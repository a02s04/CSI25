def lower_triangle(rows):
    print("\nLower Triangular Pattern:")
    for i in range(1, rows + 1):
        print("* " * i)

def upper_triangle(rows):
    print("\nUpper Triangular Pattern:")
    for i in range(rows, 0, -1):
        print("* " * i)

def pyramid(rows):
    print("\nPyramid Pattern:")
    for i in range(1, rows + 1):
        spaces = ' ' * (rows - i)
        stars = '* ' * i
        print(spaces + stars)

rows = int(input("Enter number of rows: "))
lower_triangle(rows)
upper_triangle(rows)
pyramid(rows)
