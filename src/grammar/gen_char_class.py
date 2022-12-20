print("const CHAR_CLASS: [u8; 256] = [")
for code in range(0, 256):
    l = chr(code)
    if l == ' ' or l == '\r' or l == '\n' or l=='\t':
      print("1, ", end="")
    elif l >= 'A' and l <= 'Z':
      print("2, ", end="")
    elif l >= 'a' and l <= 'z':
      print("3, ", end="")
    elif l == ':':
      print("4, ", end="")
    elif l == ';':
      print("5, ", end="")
    elif l == '|':
      print("6, ", end="")
    elif l == '[':
      print("7, ", end="")
    elif l == ']':
      print("8, ", end="")
    elif l == '(':
      print("9, ", end="")
    elif l == ')':
      print("10, ", end="")
    elif l == '*':
      print("11, ", end="")
    elif l == '+':
      print("12, ", end="")
    elif l == '?':
      print("13, ", end="")
    elif l == '~':
      print("14, ", end="")
    elif l == '.':
      print("15, ", end="")
    elif l == '\'':
      print("16, ", end="")
    elif l == '"':
      print("17, ", end="")
    elif l == '_':
      print("18, ", end="")
    elif l >= '0' and l <= '9':
      print("19, ", end="")
    elif l in {"!","#","$","%","&",",","-","/","=","<",">","@","\\","^","`","{","}"}:
      print("20, ", end="")
    else:
      print("0, ", end="")
print("];")
