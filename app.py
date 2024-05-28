import asyncio
import json
import os
import time
import pyaudio
import sys
import boto3
import sounddevice
import copy

from concurrent.futures import ThreadPoolExecutor
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent, TranscriptResultStream

from api_request_schema import api_request_list, get_model_ids

model_id = os.getenv('MODEL_ID', 'amazon.titan-text-express-v1')
aws_region = os.getenv('AWS_REGION', 'us-east-1')

if model_id not in get_model_ids():
    print(f'Error: Models ID {model_id} in not a valid model ID. Set MODEL_ID env var to one of {get_model_ids()}.')
    sys.exit(0)

api_request = api_request_list[model_id]
config = {
    'log_level': 'debug',  # One of: info, debug, none
    'last_speech': "If you have any other questions, please don't hesitate to ask. Have a great day!",
    'region': aws_region,
    'polly': {
        'Engine': 'neural',
        'LanguageCode': 'en-US',
        'VoiceId': 'Joanna',
        'OutputFormat': 'pcm',
    },
    'translate': {
        'SourceLanguageCode': 'en',
        'TargetLanguageCode': 'en',
    },
    'bedrock': {
        'response_streaming': False,
        'api_request': api_request
    }
}


p = pyaudio.PyAudio()
bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=config['region'])
polly = boto3.client('polly', region_name=config['region'])
transcribe_streaming = TranscribeStreamingClient(region=config['region'])

FUNCTION_PROMPT = '''
You will be acting as an sport coach named Joe. Your goal is to give sport advice, please keep your response under 50 characters. Don't use any exclamation point.

You have access to the following tools:
[
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city or region which is required to fetch weather information.",
                },
                "unit": {
                    "type": "string",
                    "enum": [
                        "celsius",
                        "fahrenheit"
                    ]
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "get_current_location",
        "description": "Use this tool to get the current location if user does not provide a location",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]
Please think step by step. If needed, please select one or more tools whose parameters have already been provided. Respond with only a JSON object matching the following schema inside a <json></json> xml tag:
{
    "result": "tool_use",
    "tool_calls": [
        {
            "tool": "<name of the selected tool, leave blank if no tools needed>",
            "tool_input": <parameters for the selected tool, matching the tool\'s JSON schema>
        }
    ]
    "explanation": "<The explanation why you choosed these tools.>"
}

If no further tools needed, response with only a JSON object matching the following schema:
{
    "result": "stop",
    "content": "<Your response to the user.>",
    "explanation": "<The explanation why you get the final answer.>"
}
'''

assistant_prefill = {
    'role': 'assistant',
    'content': 'Here is the result in JSON: <json>'
}


def printer(text, level):
    if config['log_level'] == 'info' and level == 'info':
        print(text)
    elif config['log_level'] == 'debug' and level in ['info', 'debug']:
        print(text)


class UserInputManager:
    shutdown_executor = False
    executor = None

    @staticmethod
    def set_executor(executor):
        UserInputManager.executor = executor

    @staticmethod
    def start_shutdown_executor():
        UserInputManager.shutdown_executor = False
        raise Exception()  # Workaround to shutdown exec, as executor.shutdown() doesn't work as expected.

    @staticmethod
    def start_user_input_loop():
        while True:
            sys.stdin.readline().strip()
            printer(f'[DEBUG] User input to shut down executor...', 'debug')
            UserInputManager.shutdown_executor = True

    @staticmethod
    def is_executor_set():
        return UserInputManager.executor is not None

    @staticmethod
    def is_shutdown_scheduled():
        return UserInputManager.shutdown_executor


class BedrockModelsWrapper:

    @staticmethod
    def define_body(text):
        NORMAL_PROMPT = "You will be acting as an sport coach named Joe. Your goal is to give sport advice, please keep your response under 50 characters. Don't use any exclamation point."
    
        model_id = config['bedrock']['api_request']['modelId']
        model_provider = model_id.split('.')[0]
        body = config['bedrock']['api_request']['body']

        if model_provider == 'amazon':
            body['inputText'] = text
        elif model_provider == 'meta':
            body['prompt'] = text
        elif model_provider == 'anthropic':
            if "claude-3" in model_id:
                #Claude3
                body['system'] = FUNCTION_PROMPT

                body['messages'] = [
                    {
                        "role": "user",
                        "content": text
                    }
                ]

                body['stop_sequences'] = ['</json>']

            else:
                #Claude2.x
                body['prompt'] = f'\n\nHuman: You will be acting as an sport coach named Joe. Your goal is to give sport advice. {text} \n\nAssistant:'
        elif model_provider == 'cohere':
            body['prompt'] = text
        else:
            raise Exception('Unknown model provider.')

        return body

    @staticmethod
    def get_stream_chunk(event):
        return event.get('chunk')

    @staticmethod
    def get_stream_text(chunk):
        model_id = config['bedrock']['api_request']['modelId']
        model_provider = model_id.split('.')[0]

        chunk_obj = ''
        text = ''
        if model_provider == 'amazon':
            chunk_obj = json.loads(chunk.get('bytes').decode())
            text = chunk_obj['outputText']
        elif model_provider == 'meta':
            chunk_obj = json.loads(chunk.get('bytes').decode())
            text = chunk_obj['generation']
        elif model_provider == 'anthropic':
            #print("model ID is :", model_id)
            if "claude-3" in model_id:
                chunk_obj = json.loads(chunk.get('bytes').decode())
                if chunk_obj['type'] == 'message_delta':
                    print(f"\nStop reason: {chunk_obj['delta']['stop_reason']}")
                    print(f"Stop sequence: {chunk_obj['delta']['stop_sequence']}")
                    print(f"Output tokens: {chunk_obj['usage']['output_tokens']}")

                if chunk_obj['type'] == 'content_block_delta':
                    if chunk_obj['delta']['type'] == 'text_delta':
                        print(chunk_obj['delta']['text'], end="")
                        text = chunk_obj['delta']['text']
            else:
                #Claude2.x
                chunk_obj = json.loads(chunk.get('bytes').decode())
                text = chunk_obj['completion']
        elif model_provider == 'cohere':
            chunk_obj = json.loads(chunk.get('bytes').decode())
            text = ' '.join([c["text"] for c in chunk_obj['generations']])
        else:
            raise NotImplementedError('Unknown model provider.')

        printer(f'[DEBUG] {chunk_obj}', 'debug')
        return text


def to_audio_generator(bedrock_stream):
    prefix = ''
    streaming = False
    if streaming:
        print("Milestone audio genrator #1")
        if bedrock_stream:
            print("Milestone audio genrator #2")
            for event in bedrock_stream:
                print("Milestone audio genrator #3")
                chunk = BedrockModelsWrapper.get_stream_chunk(event)
                printer('[DEBUG] chunk: {chunk}', 'debug')
                if chunk:
                    #print("chunk is :", chunk)
                    text = BedrockModelsWrapper.get_stream_text(chunk)
                    print("text is :", text)
                    if '.' in text or '!' in text:
                        print("Detect the entire sentence")
                        a = text.split('.')[:-1]
                        to_polly = ''.join([prefix, '.'.join(a), '. '])
                        prefix = text.split('.')[-1]
                        print(to_polly, flush=True, end='')
                        #added by Teague
                        #aws_polly_tts(to_polly)
                        yield to_polly
                    else:
                        prefix = ''.join([prefix, text])

            if prefix != '':
                print(prefix, flush=True, end='')
                yield f'{prefix}.'

            print('\n')
    else:
        if bedrock_stream:
            print("bedrock stream is: \n", bedrock_stream)
            for text in bedrock_stream:
                #print("chunk is :", chunk)
                print("Milestone audio genrator #3")
                #print("text is :", text)
                if '.' in text or '!' in text:
                    print("Detect the entire sentence")
                    a = text.split('.')[:-1]
                    to_polly = ''.join([prefix, '.'.join(a), '. '])
                    prefix = text.split('.')[-1]
                    print(to_polly, flush=True, end='')
                    yield to_polly
                else:
                    prefix = ''.join([prefix, text])

            if prefix != '':
                print(prefix, flush=True, end='')
                yield f'{prefix}.'

            print('\n')

def get_current_location():
    # Mock response
    return 'Guangzhou'

def get_current_weather(location, unit='celsius'):
    # Mock response
    print(f'location: {location}')
    if location == 'Guangzhou':
        return 'Guangzhou: Sunny at 25 degrees Celsius.'
    elif location == 'Beijing':
        return ' Beijing: Rainy at 30 degrees'
    return 'It\'s a normal sunny day~'

function_map = {
    'get_current_location': get_current_location,
    'get_current_weather': get_current_weather
}

def complete(body):

    model_id = 'anthropic.claude-3-haiku-20240307-v1:0'
    """
    body=json.dumps(
        {
            'anthropic_version': 'bedrock-2023-05-31',
            'max_tokens': 1000,
            'system': FUNCTION_PROMPT,
            'temperature': 0,
            'messages': [*messages, assistant_prefill],
            'stop_sequences': ['</json>']
        }
    )
    """
    #body_copy = copy.deepcopy(body)
    body['messages'].append(assistant_prefill)

    print("\nMessages after prefill is \n", body)
    body = json.dumps(body)
    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)

    #removing assistant prefill after inferencing
    #body = copy.deepcopy(body_copy)
    print("body_copy is \n", body)

    print("Agent is inferencing")
    response_body = json.loads(response.get('body').read())
    print(response_body)
    text = response_body['content'][0]['text']
    print(text)
    return parse_json_str(text)

def stream_complete(body):
    model_id = 'anthropic.claude-3-haiku-20240307-v1:0'
    """
    body=json.dumps(
        {
            'anthropic_version': 'bedrock-2023-05-31',
            'max_tokens': 1000,
            'system': FUNCTION_PROMPT,
            'temperature': 0,
            'messages': [*messages, assistant_prefill],
            'stop_sequences': ['</json>']
        }
    )
    """
    body['messages'].append(assistant_prefill)

    print("\nMessages after prefill is \n", body)
    body = json.dumps(body)

    response = bedrock_runtime.invoke_model_with_response_stream(
        body=body, modelId=model_id
    )

    '''
    result_chunks = ''
    print('LLM Response: \n')
    for event in response.get("body"):
        chunk = json.loads(event["chunk"]["bytes"])

        if chunk['type'] == 'content_block_delta' and chunk['delta']['type'] == 'text_delta':
            text = chunk['delta']['text']
            print(text, end='')
            result_chunks += text
    return parse_json_str(result_chunks)
    '''
    return response


def agents(body, stream=True):
    print("\nAgent Milstone #1")
    body_init = copy.deepcopy(body)
    finished = False
    response = ''
    while not finished:
        result = {}
        print("body tracker #1\n", body)
        result_stream = stream_complete(body)
        result_chunks = ''
        print('LLM Response: \n')
        for event in result_stream.get("body"):
            chunk = json.loads(event["chunk"]["bytes"])

            if chunk['type'] == 'content_block_delta' and chunk['delta']['type'] == 'text_delta':
                text = chunk['delta']['text']
                print(text, end='')
                result_chunks += text
        result = parse_json_str(result_chunks)

        body = copy.deepcopy(body_init)
        print("body tracker #2\n", body)

        print("\nAgent Milstone #2")

        if result['result'] == 'tool_use':
            assistant_msg = ''
            function_msg = ''
            for t in result['tool_calls']:
                tool = t['tool']
                tool_input = t['tool_input']
                assistant_msg += f'Should use {tool} tool with args: {json.dumps(tool_input)}\n'
                function2call = function_map[tool]
                # calling the function
                function_result = function2call(**tool_input)
                # Append to prompts
                function_msg += f'I have used the {tool} tool with args: {json.dumps(tool_input)} and the result is : {function_result}\n'
            
            body = body_init
            print(type(body))
            print(body['messages'])
            #body=json.load(body)
            print(type(body))
            body['messages'].append({'role': 'assistant', 'content': assistant_msg})
            body['messages'].append({'role': 'user', 'content': function_msg})
            print(body)
        elif result['result'] == 'stop':
            finished = True
            response = result['content']
            
    return response, result_stream

def parse_json_str(json_str):
    # response from LLM may contains \n
    result = {}
    try:
        result = json.loads(json_str.replace('\n', '').replace('\r', ''))
        print('LLM response can be parsed as a valid JSON object.')
    except Exception as e:
        print('Cannot parsed to a valid python dict object')
        print(e)
    return result



class BedrockWrapper:

    def __init__(self):
        self.speaking = False

    def is_speaking(self):
        return self.speaking

    def invoke_bedrock(self, text):
        printer('[DEBUG] Bedrock generation started', 'debug')
        self.speaking = True

        body = BedrockModelsWrapper.define_body(text)
        printer(f"[DEBUG] Request body: {body}", 'debug')

        try:
            #body_json = json.dumps(body)
            #print('body_json: \n', body_json)
            """
            response = bedrock_runtime.invoke_model_with_response_stream(
                body=body_json,
                modelId=config['bedrock']['api_request']['modelId'],
                accept=config['bedrock']['api_request']['accept'],
                contentType=config['bedrock']['api_request']['contentType']
            )
            print("Using the original response")
            """
            print('\nbody: \n', body)
            #body = json.dumps(body)
            response, bedrock_stream = agents(body)
            print("Using the function calling")

            #print("Response is \n",response)
            
            

            printer('[DEBUG] Capturing Bedrocks response/bedrock_stream', 'debug')
            #bedrock_stream = result_stream.get('body')
            #bedrock_stream = response
            print("bedrock_stream milestone#1")
            print("response is \n", response)
            print("bedrock_stream is \n", bedrock_stream)

            #for debuging
            '''
            result_chunks = ''
            print('LLM Response: \n')
            for event in bedrock_stream.get("body"):
                print("event:\n", event)
                chunk = json.loads(event["chunk"]["bytes"])
                print("chunk is \n", chunk)
                if chunk['type'] == 'content_block_delta' and chunk['delta']['type'] == 'text_delta':
                    text = chunk['delta']['text']
                    print(text, end='')
                    result_chunks += text
            print ("result_chunks is \n",result_chunks)
            '''

            #audio_gen = to_audio_generator(bedrock_stream)
            #Try using the entire response string instead of streaming response
            audio_gen = to_audio_generator(response)
            printer('[DEBUG] Created bedrock stream to audio generator', 'debug')

            reader = Reader()
            for audio in audio_gen:
                reader.read(audio)

            reader.close()

        except Exception as e:
            print(e)
            time.sleep(2)
            self.speaking = False

        time.sleep(1)
        self.speaking = False
        printer('\n[DEBUG] Bedrock generation completed', 'debug')


class Reader:

    def __init__(self):
        self.polly = boto3.client('polly', region_name=config['region'])
        self.audio = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)
        self.chunk = 1024

    def read(self, data):
        response = self.polly.synthesize_speech(
            Text=data,
            Engine=config['polly']['Engine'],
            LanguageCode=config['polly']['LanguageCode'],
            VoiceId=config['polly']['VoiceId'],
            OutputFormat=config['polly']['OutputFormat'],
        )

        stream = response['AudioStream']

        while True:
            # Check if user signaled to shutdown Bedrock speech
            # UserInputManager.start_shutdown_executor() will raise Exception. If not ideas but is functional.
            if UserInputManager.is_executor_set() and UserInputManager.is_shutdown_scheduled():
                UserInputManager.start_shutdown_executor()

            data = stream.read(self.chunk)
            self.audio.write(data)
            if not data:
                break

    def close(self):
        time.sleep(1)
        self.audio.stop_stream()
        self.audio.close()


def stream_data(stream):
    chunk = 1024
    if stream:
        polly_stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            output=True,
        )

        while True:
            data = stream.read(chunk)
            polly_stream.write(data)

            # If there's no more data to read, stop streaming
            if not data:
                time.sleep(0.5)
                stream.close()
                polly_stream.stop_stream()
                polly_stream.close()
                break
    else:
        # The stream passed in is empty
        pass


def aws_polly_tts(polly_text):
    printer(f'[INTO] Character count: {len(polly_text)}', 'debug')
    byte_stream_list = []
    polly_text_len = len(polly_text.split('.'))
    printer(f'LEN polly_text_len: {polly_text_len}', 'debug')
    for i in range(0, polly_text_len, 20):
        printer(f'{i}:{i + 20}', 'debug')
        polly_text_chunk = '. '.join(polly_text.split('. ')[i:i + 20])
        printer(f'polly_text_chunk LEN: {len(polly_text_chunk)}', 'debug')

        response = polly.synthesize_speech(
            Text=polly_text_chunk,
            Engine=config['polly']['Engine'],
            LanguageCode=config['polly']['LanguageCode'],
            VoiceId=config['polly']['VoiceId'],
            OutputFormat=config['polly']['OutputFormat'],
        )
        byte_stream = response['AudioStream']
        byte_stream_list.append(byte_stream)

    byte_chunks = []
    chunk = 1024
    for bs in byte_stream_list:
        while True:
            data = bs.read(chunk)
            byte_chunks.append(data)

            if not data:
                bs.close()
                break

    read_byte_chunks(b''.join(byte_chunks))


def read_byte_chunks(data):
    polly_stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)
    polly_stream.write(data)

    time.sleep(1)
    polly_stream.stop_stream()
    polly_stream.close()
    time.sleep(1)


class EventHandler(TranscriptResultStreamHandler):
    text = []
    last_time = 0
    sample_count = 0
    max_sample_counter = 4

    def __init__(self, transcript_result_stream: TranscriptResultStream, bedrock_wrapper):
        super().__init__(transcript_result_stream)
        self.bedrock_wrapper = bedrock_wrapper

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        if not self.bedrock_wrapper.is_speaking():

            if results:
                for result in results:
                    EventHandler.sample_count = 0
                    if not result.is_partial:
                        for alt in result.alternatives:
                            print(alt.transcript, flush=True, end=' ')
                            EventHandler.text.append(alt.transcript)

            else:
                EventHandler.sample_count += 1
                if EventHandler.sample_count == EventHandler.max_sample_counter:

                    if len(EventHandler.text) == 0:
                        last_speech = config['last_speech']
                        print(last_speech, flush=True)
                        #aws_polly_tts(last_speech)
                        #os._exit(0)  # exit from a child process
                    else:
                        input_text = ' '.join(EventHandler.text)
                        printer(f'\n[INFO] User input: {input_text}', 'info')

                        executor = ThreadPoolExecutor(max_workers=1)
                        # Add executor so Bedrock execution can be shut down, if user input signals so.
                        UserInputManager.set_executor(executor)
                        loop.run_in_executor(
                            executor,
                            self.bedrock_wrapper.invoke_bedrock,
                            input_text
                        )

                    EventHandler.text.clear()
                    EventHandler.sample_count = 0


class MicStream:

    async def mic_stream(self):
        loop = asyncio.get_event_loop()
        input_queue = asyncio.Queue()

        def callback(indata, frame_count, time_info, status):
            loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))

        stream = sounddevice.RawInputStream(
            channels=1, samplerate=16000, callback=callback, blocksize=2048 * 2, dtype="int16")
        with stream:
            while True:
                indata, status = await input_queue.get()
                yield indata, status

    async def write_chunks(self, stream):
        async for chunk, status in self.mic_stream():
            await stream.input_stream.send_audio_event(audio_chunk=chunk)

        await stream.input_stream.end_stream()

    async def basic_transcribe(self):
        loop.run_in_executor(ThreadPoolExecutor(max_workers=1), UserInputManager.start_user_input_loop)

        stream = await transcribe_streaming.start_stream_transcription(
            language_code="en-US",
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )

        handler = EventHandler(stream.output_stream, BedrockWrapper())
        await asyncio.gather(self.write_chunks(stream), handler.handle_events())


info_text = f'''
*************************************************************
[INFO] Supported FM models: {get_model_ids()}.
[INFO] Change FM model by setting <MODEL_ID> environment variable. Example: export MODEL_ID=meta.llama2-70b-chat-v1

[INFO] AWS Region: {config['region']}
[INFO] Amazon Bedrock model: {config['bedrock']['api_request']['modelId']}
[INFO] Polly config: engine {config['polly']['Engine']}, voice {config['polly']['VoiceId']}
[INFO] Log level: {config['log_level']}

[INFO] Hit ENTER to interrupt Amazon Bedrock. After you can continue speaking!
[INFO] Go ahead with the voice chat with Amazon Bedrock!
*************************************************************
'''
print(info_text)

loop = asyncio.get_event_loop()
try:
    loop.run_until_complete(MicStream().basic_transcribe())
except (KeyboardInterrupt, Exception) as e:
    print()
