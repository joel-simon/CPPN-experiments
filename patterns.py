import numpy as np

def pretty_print(x):
    """
    0 : Black
    1 : Green
    2 : Yellow
    3 : Red
    4 : Empty
    """
    # 91 = red
    colors = [ "\033[90m _", "\033[92m #", "\033[93m #", "\033[93m #", "\033[90m  "]
    fn = lambda i: colors[int(i)]

    for r in x:
        print(''.join(map(fn, r)))

    print("\033[00m")

def checkerboard(width, height, block_size):
    board = np.zeros((width, height), dtype=bool)
    for i in range(0, height, block_size):
        for j in range(0, width, block_size*2):
            if i % (2*block_size) == 0:
                board[i: i+block_size, j: j+block_size] = 1
            else:
                board[i: i+block_size, j+block_size: j+2*block_size] = 1
    return board

def random(width, height):
    return np.random.randint(0, 2, (height, width))

def random_scaled(width, height, scale=3):
    a = np.random.randint(0, 2, (height//scale, width/scale))
    b = np.zeros((width, height), dtype=int)
    for (i, j), v in np.ndenumerate(a):
        b[i*scale:i*scale+scale, j*scale:j*scale+scale] = v
    return b

if __name__ == '__main__':
    pretty_print(random(32, 32))
    pretty_print(random_half_scaled(32, 32))
