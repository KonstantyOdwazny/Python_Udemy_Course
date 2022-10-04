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
user_iq = 190
age = user_iq / 5  #  user_iq / 5 - expression, a calosc to statement
# _user_iq_ = 190 # Oznacza prywatna zmienna
a, b, c = 1, 2, 3
print(a)
print(b)
print(c)

# augmented assignment operator
some_value = 5
some_value = 5+2
some_value = some_value + 2
some_value += 2
some_value -= 2
some_value *= 2
some_value /= 2

print(some_value)
