from myfunc import y

def main():
    xs = [0, 0.5, 1.0, 2.0, 3.0]
    print("y(x) values:")
    for x in xs:
        val = y(x)
        print(f"x={x:.6f}, y(x)={val:.6f}")

if __name__ == "__main__":
    main()
