import cauldron as cd

print('sharing worked?')
print(cd.shared.x)
tester = cd.shared.fetch('x')

print('\n')
print(tester)
