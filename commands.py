from config import get_config

from example_package import show_message

def example_message():
    '''
    This is a test function showing an example message.
    '''
    message_text = get_config('message', 'text')
    message_number = get_config('message', 'number')
    show_message(message_text, message_number=message_number)
