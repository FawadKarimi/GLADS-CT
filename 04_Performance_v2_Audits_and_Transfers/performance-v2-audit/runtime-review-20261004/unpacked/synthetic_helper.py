"""Finite synthetic payload generator; no native operations or live entry point."""


def blocks(scenario):
    if scenario == 'success':
        yield b'pv2-native-synthetic-v1\n'
    elif scenario == 'output_overflow':
        for _ in range(17):
            yield b'x' * 65536
    elif scenario == 'timeout':
        # A fake scheduler models blocking. Never sleep or spin in this module.
        raise TimeoutError('synthetic helper timeout scenario')
    else:
        raise ValueError('unknown synthetic scenario')
