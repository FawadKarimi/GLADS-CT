"""Independent bounded finalization; first nonzero execution status is immutable."""
def finalize(status, steps):
    errors=[]
    for stage,call in steps:
        try:call()
        except BaseException as error:
            errors.append({'stage':stage,'category':type(error).__name__})
    return (status if status else (1 if errors else 0)),errors
