# name = input('What is your name?\n')
# print(f'Hello {name}!')
# Fundamental Data Types:
# a) Numbers:
# int, float
# print(2+4," Type:", type(2+4)) #int
# print(2-4," Type:", type(2-4)) # int
# print(2*4," Type:", type(2*4)) # int
# print(2/4, " Type:", type(2/4)) # float
# print( 2**2) # ** potega
# print( 6 // 4) # zwraca integer z dzielenia
# print( 6 % 4) # operacja modulo (reszta z dzielenia)

# Math functions:
# print(round(3.5)) # zaokraglanie
# print(abs(-20))

# operator precedence - pierwszenstwo operatora
#1. ()
#2. **
#3. * and /
#4. + and -
# print((20-3)+5 - 3*4)

# Optional bin() and complex
# print(bin(5)) # zamiana liczb na liczby binarne
# print(int('0b101',2)) # zamiana liczb binarnych na dziesietne

# 2. Variabels
# user_iq = 190
# age = user_iq / 5  #  user_iq / 5 - expression, a calosc to statement
# # _user_iq_ = 190 # Oznacza prywatna zmienna
# a, b, c = 1, 2, 3
# print(a)
# print(b)
# print(c)

# # augmented assignment operator
# some_value = 5
# some_value = 5+2
# some_value = some_value + 2
# some_value += 2
# some_value -= 2
# some_value *= 2
# some_value /= 2

# print(some_value)

#String
# username = 'supercoder'
# password = "supersecret"
# long_string = '''
# WOW
# 0 0
# ---
# '''
# print(long_string)

# first_name = 'Andrei'
# last_name = 'Neagoie'
# full_name = first_name +" "+ last_name
# print(full_name)

# # string concatenation - powiązanie
# # string mozna dodac tylko do stringa

# #Type conversion

# print("Hello " + str(100))

# Escape Sequence
# weather = "It\'s \n \"kind of\"\t sunny \\"

# print(weather)

# formatted strings
# name = 'Jonny'
# age = 55

# print('Hi '+name +'!. You are '+ str(age) + ' years old ')
# # Nowosc w Python 3 (dodanie f przed stringiem)
# print(f'Hi {name}!. You are {age} years old ') # - F string
# print('Hi {}!. You are {} years old '.format(name, age))
# print('Hi {new_name}!. You are {new_age} years old '.format(new_name = 'Sally', new_age=35))

# String Indexes
# text = '0123456'
# print(text[0])
# # [start:stop:stepover]
# print(text[0:6])
# print(text[0:6:2])
# print(text[1:])
# print(text[:4])
# print(text[::2])
# print(text[-1])
# # Odwrocenie stringa
# print(text[::-1])
# print(text[::-2])

# Immutability - niezmiennosc
# selfish = '01234567'

# # selfish= 100 # moge zmiennic zmienna na inta

# # selfish[0] = '8' # nie moge jednak zmienic czegos wewnatrz stringa

# selfish += '8' # mozemy jednak dodawac nowe wartosci do stringa

# print(selfish)

# Password checker
# username = input('Login: ')
# password = input('Password: ')

# print(f"{username} your password {'*' *len(password)} is {len(password)} letters long" )


# List slicing 
# amazon_chart = ['noteboks',
#                'Sunglass',
#                10,
#                'graps',
#                23]
# amazon_chart[0] = 'laptop'
# # new_cart = amazon_chart # jezeli tak zrobie to jak zmienie new_cart zmieni sie
# #tez amazon cart poniewaz to ta sama zmienna teraz
# new_cart = amazon_chart[:] # teraz jest to kopia ale nie zmienia juz oryginalu
# new_cart[0] = 'gum'
# print(amazon_chart[0::2])
# print(new_cart)

# List method 
# basket = ['a','b','d','c','a','x','f','e']

# # basket.sort() # zmienia nam nasza liste na posortowana
# # print(basket)

# print(sorted(basket)) # tylko wyswietla posortowana liste nie zminiajac jej
# print(basket)
# basket.reverse() # odwraca cala liste
# print(basket) 

# Common list patterns
# sentence = ' '
# new = sentence.join(['hi','my','name','is', 'Kostek']) # rozlozenie listy na string

# # lub
# new_s = ' '.join(['hi','my','name','is', 'Kostek'])

# print(new_s)

# Dictionary
# dictionary = {   # slownik to nieuporzadkowany zboir kluczy i wartosci
#   'a':1,
#   'b':2
# }

# print(dictionary['b'])

# # Jezeli chcemy sprawdzic czy jakis klucz istnieje w naszym slowniku to 
# # to uzywamy .get() zeby nie bylo bledu
# print(dictionary.get('s'))

# # Mozemy tez w get ustawic wartosc defult dla klucza, w przypadku gdyby nie istnial
# print(dictionary.get('s',22))
# # Ale gdy istnieje to wartosc default nic nie zmienia
# print(dictionary.get('a', 122))

# Tuple - jest immutable (nie mozna go modyfikowac) 
# to taka lista ktorej nie mozna modyfikowac

# my_tuple = (1,2,3,4,5,6)
# # my_tuple[2] = 9 nie moge tak zrobic
# my_list = [1,2,3,4,5,6]
# my_list[2] = 9 # z lista moge tak zrobic
# print(my_tuple[2])
# print(my_list[2])

# Set
# To kolelka nieuporzadkowanych unikalnych wartosci
my_set = {1,2,2,2,3,3,4,5,6,9,0}
my_set.add(100)
my_set.add(9)
print(my_set)
my_list = [1,2,3,3,3,3,4,6,7]
# konwersja lista na set
new_set = set(my_list)
print(my_list)
print(new_set)

