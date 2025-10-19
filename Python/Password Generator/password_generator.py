import random

print('\n')
upper_alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
lower_alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
numbers = ['0','1','2','3','4','5','6','7','8','9']
symbols = ['*','/','.','?','[]',';','@','!','$','#','^','%','&','+','-']

positive_response = ['y','yes','ok']
negative_response = ['n','no']

sym_include = input("Include Symbols? (y/n) ").lower()

if sym_include in positive_response:	
	characters = upper_alphabet + lower_alphabet + numbers + symbols
elif sym_include in negative_response:
	characters = upper_alphabet + lower_alphabet + numbers

print('\n')

length = int(input('Length of password: '))
password = ""
for i in range(length):
	password=password + random.choice(characters)
print('\n')
print(password)
