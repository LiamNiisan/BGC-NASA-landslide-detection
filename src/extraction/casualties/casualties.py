from word2number import w2n


def format_num(number):
    """
    Transforms numbers from if they are in a textual form
    to a numerical form.

    Parameters
    ----------
    number : str
        number to transform

    Returns
    -------
    str
        transformed number
    """    
    if number.isdigit():
        return number
    else:
        try:
            return w2n.word_to_num(number.lower())
        except ValueError:
            return ""


def is_num(token):
    """
    Checks if a certain string is a number or not.

    Parameters
    ----------
    token : str
        token to check.

    Returns
    -------
    bool
        number or not.
    """    
    if token.isdigit():
        return True
    else:
        try:
            w2n.word_to_num(token.lower())
            return True
        except ValueError:
            return False
