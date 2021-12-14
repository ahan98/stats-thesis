def bitmasks(n, m):
    if n == m:
        yield (1 << n) - 1
    elif m > 0:
        for x in bitmasks(n-1, m-1):
            yield (1 << (n-1)) + x
        for x in bitmasks(n-1, m):
            yield x
    else:
        yield 0


if __name__ == "__main__":
    print(list(bitmasks(5, 3)))
