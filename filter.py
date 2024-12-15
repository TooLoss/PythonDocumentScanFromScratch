def WhiteScale(I, p):
    x, y = I.shape
    low, high = I.min()/p, I.max()/p
    for i in range(x):
        for j in range(y):
            I[i,j] = 255*(I[i,j]-low)/(high-low)
    return I