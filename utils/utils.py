def write_print(path, text):
    """Displays text in console and saves in text file

    [description]

    Arguments:
        path {string} -- path to text file
        text {string} -- text to display and save
    """
    file = open(path, 'a')
    file.write(text + '\n')
    file.close()
    print(text)
