def show_message(message_text, message_number=1):
    print('Starting command "show_message"...')
    print(
        'The message was:\n' +
        '\n'.join([message_text] * message_number)
    )
