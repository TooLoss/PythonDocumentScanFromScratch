import fontforge

char_list = [chr(c) for c in range(ord('A'), ord('Z') + 1)] + [chr(c) for c in range(ord('a'), ord('z') + 1)]

#########

path = 'font/Aptos.ttf'
fontname = "Aptos"

#########

F = fontforge.open(path)
for name in char_list:
    filename = str(name) + str(fontname) + ".png"
    if ord(name) in F:
        F[ord(name)].export('./font/export/' + filename, 128)
    else:
        print(str(ord(name)) + " not found")