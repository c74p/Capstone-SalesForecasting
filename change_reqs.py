with open('./requirements_change.txt', 'r') as ref:
    with open('/home/chachi/requirements.txt', 'w') as target:
        for line in ref:
            pkg, *args = line.strip().split(' ')
            ver = ""
            for arg in args:
                if arg != "":
                    ver = arg
            print(pkg + '=' + ver)
            target.write(pkg + '=' + ver + '\n')
