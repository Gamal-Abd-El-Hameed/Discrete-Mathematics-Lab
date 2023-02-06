import numpy as np
import math
from tabulate import tabulate
import random

primeNumbers = list([])
class Eratosthenes:
    def __init__(self, number):
        self.integerNumbers = list(np.linspace(2, number, number - 1, dtype = int))
        #self.divisors = list([2, 3, 5, 7])
        self.notPrimeNumbers = []
        #Run the algorithm with the divisors "2, 3, 5, 7"
        self.computEratosthenes(2)
        self.computEratosthenes(3)
        self.computEratosthenes(5)
        self.computEratosthenes(7)
        self.returnPrimeNumbers()
        
    def computEratosthenes(self, number):
        for i in range(len(self.integerNumbers)):
            remainder = self.integerNumbers[i] % number
            """
            if remainder equal to zero it means it is not a prime number
            put this number in array of notPrimeNumbers 
            """
            if(remainder == 0 and self.integerNumbers[i] != number): 
                self.notPrimeNumbers.append(self.integerNumbers[i])
        self.removeNotPrimeNumbers()
    #remove notPrimeNumbers from the original array
    def removeNotPrimeNumbers(self):
        for i in range(len(self.notPrimeNumbers)):
            self.integerNumbers.remove(self.notPrimeNumbers[i]) 
        self.notPrimeNumbers = list([])
    
    def returnPrimeNumbers(self):
        for i in range(len(self.integerNumbers)):
            primeNumbers.append(self.integerNumbers[i])
 
class TrialDivision(Eratosthenes):
    def __init__(self, number):
        self.initailNumber = number
        self.inputNumber = int(number)
        """
        Trial division algorithm use the inheritance concept
        it call Eratosthenes algorithm to get the prime numbers under the square root
        """
        self.sqrt = int(math.sqrt(self.inputNumber))
        Eratosthenes(self.sqrt)
        self.result = list([])
        self.computeFactors()
        self.printResult()
    def computeFactors(self):
        counter2 = 0 #index of primeNumbers
        while(counter2 <= len(primeNumbers) - 1):  
            divisor = int(primeNumbers[counter2])
            divisionResult = self.inputNumber % divisor
            if(divisionResult == 0):
                self.result.append(divisor)
                self.inputNumber /= divisor 
            else:
                counter2 += 1
            
    def printResult(self):
        self.checkNumbers()
        if(len(self.result) == 0):
            print(self.inputNumber)
        else:
            print(self.result)
            
    def checkNumbers(self):
        res = 1
        for i in range(len(self.result)):
            res *= self.result[i]
        if(res != self.initailNumber): 
            self.result.append(self.initailNumber // res)




class ChineseRemainder:
    def __init__(self):
        self.takeInput()
        self.computeCapital_M()
        self.comput_Mi()
        self.chineseRemainderAlgorithm()
        self.computeResult()
    
    def takeInput(self):
        self.numberOfEquations = abs(int(input("Number of Equations: ")))
        #2D array for the computations
        """
        x[0] = b
        x[1] = m
        x[2] = Mi  e.g. M1 = M/m1
        x[3] = xi
        """
        self.inputMatix = list([[0 for i in range(self.numberOfEquations + 1)]
                               for j in range(self.numberOfEquations)])
        print("Equation form x≡b(mod m)")
        for i in range(self.numberOfEquations):
            print("Params of Eqn ", i + 1, ":")
            self.inputMatix[i][0] = abs(int(input("b: ")))
            self.inputMatix[i][1] = abs(int(input("m: ")))
            
    def computeCapital_M(self):
        self.M = 1
        for i in range(self.numberOfEquations):
            self.M *= self.inputMatix[i][1]
    
    def comput_Mi(self):
        for i in range(self.numberOfEquations):
            self.inputMatix[i][2] = self.M / self.inputMatix[i][1]
            
    def chineseRemainderAlgorithm(self):
        for i in range(len(self.inputMatix)):
            remaider = self.inputMatix[i][2] % self.inputMatix[i][1]
            #if remainder equals 1 it means the number is the inverse
            if(remaider == 1):
                self.inputMatix[i][3] = 1
            else:
                flag = True
                counter = 2
                while(flag):
                    temp = (counter * remaider) % self.inputMatix[i][1]
                    if(temp == 1):
                        flag = False
                        self.inputMatix[i][3] = counter
                    counter += 1
                    
    def computeResult(self):
        self.result = 0
        for i in range(self.numberOfEquations):
            self.result += self.inputMatix[i][0] * self.inputMatix[i][2] * self.inputMatix[i][3]
        print("X = " + str(int(self.result % self.M)) + " + " + str(self.M) + "n")


class Miller_Test:
    def __init__(self):
        self.takeInput()
        self.printResults()

    def takeInput(self):
        self.n = abs(int(input("Enter the Number: ")))
        try:
            self.ITERATIONS = abs(int(input("Enter Number of Iterations: ")))
        except ValueError:
            self.ITERATIONS = 30  # Default value

    # (x ^ y) % p
    def modularPower(self, x, y):
        ans = 1
        x %= self.n
        while (y > 0):
            if (y & 1):  # If y is odd
                ans = (ans * x) % self.n

            y = y >> 1  # Divide y by 2
            x = (x * x) % self.n
        return ans

    # returns false if n is composite
    # and true if n is probably prime
    def MillerTest(self):
        # n - 1 = t * (2 ^ s)
        # s >= 0, t > 0 & t is odd

        # Choose a random base in [2, n-2]
        self.b = 2 + random.randint(1, self.n - 4)

        # x = (b ^ t) % n
        self.x = self.modularPower(self.b, self.t)

        # n is probably prime if
        # (b ^ t) % n = 1 or
        # (b ^ t) % n = (n - 1)
        if (self.x == 1 or self.x == self.n - 1):
            return True

        while (self.t != self.n - 1):
            self.x = (self.x * self.x) % self.n
            self.t *= 2

            # n is probably prime if
            # (x ^ 2) % n = (n - 1)
            if (self.x == self.n - 1):
                return True

            if (self.x == 1):
                return False

        return False

    def isPrime(self):
        if (self.n == 2 or self.n == 3):
            return True  # 2 & 3 are Primes

        if (self.n == 1 or (self.n % 2 == 0)):
            return False  # 1 & even numbers greater than 2 are composites

        self.t = self.n - 1  # n is odd, so t must be even
        while (self.t % 2 == 0):  # we need t to be odd
            self.t //= 2

        for iteration in range(self.ITERATIONS):
            if (self.MillerTest() == False):
                return False

        return True

    def printResults(self):
        if (self.isPrime()):
            print("Prime")
        else:
            print("Composite")


class Extended_Euclidean:
    def __init__(self):
        self.initializeVariables()
        self.takeInput()
        self.setFirstRow()
        self.is_B_factor_of_A()
        self.setSecondRow()
        self.setOtherRows()
        self.getBezoutCoffiecients()
        self.printResults()

    def initializeVariables(self):
        self.A = list()  # 2D Array of the result table
        self.counter = 0  # Index of the row in the table
        self.INDEX = 0   # Index of the 1st column in the table (j)
        self.RJ = 1      # Index of the 2nd column in the table (r(j))
        self.RJ1 = 2     # Index of the 3rd column in the table (r(j + 1))
        self.QJ1 = 3     # Index of the 4th column in the table (q(j + 1))
        self.RJ2 = 4     # Index of the 5th column in the table (r(j + 2))
        self.SJ = 5      # Index of the 6th column in the table (s(j))
        self.TJ = 6      # Index of the 7th column in the table (t(j))
        self.NumberOfColumns = 7
        self.flag = True  # false if one number is multiple of the other

    def takeInput(self):
        self.A.append(list(range(self.NumberOfColumns)))
        self.a = abs(int(input("Enter the First Number: ")))
        self.b = abs(int(input("Enter the Second Number: ")))

    # put the values of the first row
    def setFirstRow(self):
        self.A[self.counter][self.INDEX] = self.counter  # Number of the row

        self.A[self.counter][self.RJ] = self.a  # r0 = a
        self.A[self.counter][self.RJ1] = self.b  # r1 = b

        # q(j + 1) = floor(a / b)
        self.A[self.counter][self.QJ1] = self.A[self.counter][self.RJ] // self.A[self.counter][self.RJ1]

        # r(j + 2) = a % b
        self.A[self.counter][self.RJ2] = self.A[self.counter][self.RJ] % self.A[self.counter][self.RJ1]

        # Bezout’s coefficients are 1 & 0
        self.A[self.counter][self.SJ] = 1
        self.A[self.counter][self.TJ] = 0
        self.remainder = self.A[self.counter][self.RJ2]  # the remainder

    # Check if one number is multiple of the other
    def is_B_factor_of_A(self):
        if(self.remainder == 0):
            self.flag = False  # finish the program
            self.gcd = self.b  # The GCD of the two numbers
            # Add another row to get the Bezout’s coefficients
            self.counter += 1
            self.A.append(list(range(self.NumberOfColumns)))
            self.A[self.counter][self.INDEX] = self.counter
            self.A[self.counter][self.RJ] = ''
            self.A[self.counter][self.RJ1] = ''
            self.A[self.counter][self.QJ1] = ''
            self.A[self.counter][self.RJ2] = ''
            self.A[self.counter][self.SJ] = 0
            self.A[self.counter][self.TJ] = 1

            # Bezout’s coefficients are 0 & 1
            self.s = 0
            self.t = 1

    # put the values of the second row
    def setSecondRow(self):
        # if the two numbers are not multiple of each other
        if(self.remainder != 0):
            # Add a new row
            self.counter += 1
            self.A.append(list(range(self.NumberOfColumns)))

            # Number of the row
            self.A[self.counter][self.INDEX] = self.counter

            # get the values of r(j) & r(j + 1) from the previous row
            self.A[self.counter][self.RJ] = self.A[self.counter - 1][self.RJ1]
            self.A[self.counter][self.RJ1] = self.A[self.counter - 1][self.RJ2]

            # q(j + 1) = floor(r(j) / r(j + 1))
            self.A[self.counter][self.QJ1] = self.A[self.counter][self.RJ] // self.A[self.counter][self.RJ1]

            # r(j + 2) = r(j) % r(j + 1)
            self.A[self.counter][self.RJ2] = self.A[self.counter][self.RJ] % self.A[self.counter][self.RJ1]

            # Bezout’s coefficients are 0 & 1
            self.A[self.counter][self.SJ] = 0
            self.A[self.counter][self.TJ] = 1
            self.remainder = self.A[self.counter][self.RJ2]  # the remainder

    # put the values of the other rows
    def setOtherRows(self):
        # the same as the second row except the Bezout’s coefficients
        while(self.remainder != 0):
            self.counter += 1
            self.A.append(list(range(self.NumberOfColumns)))
            self.A[self.counter][self.INDEX] = self.counter
            self.A[self.counter][self.RJ] = self.A[self.counter - 1][self.RJ1]
            self.A[self.counter][self.RJ1] = self.A[self.counter - 1][self.RJ2]
            self.A[self.counter][self.QJ1] = self.A[self.counter][self.RJ] // self.A[self.counter][self.RJ1]
            self.A[self.counter][self.RJ2] = self.A[self.counter][self.RJ] % self.A[self.counter][self.RJ1]

            # Bezout’s coefficients
            # s(j) = s(j - 2) - q(j - 1) * s(j - 1)
            self.A[self.counter][self.SJ] = self.A[self.counter - 2][self.SJ] - \
                self.A[self.counter - 2][self.QJ1] * \
                self.A[self.counter - 1][self.SJ]

            # t(j) = t(j − 2) − q(j − 1) * t(j − 1)
            self.A[self.counter][self.TJ] = self.A[self.counter - 2][self.TJ] - \
                self.A[self.counter - 2][self.QJ1] * \
                self.A[self.counter - 1][self.TJ]

            self.remainder = self.A[self.counter][self.RJ2]  # the remainder

    def getBezoutCoffiecients(self):
        if(self.flag):
            self.gcd = self.A[self.counter][self.RJ1]  # GCD of the two numbers

            # Add another row to get the Bezout’s coefficients
            self.counter += 1
            self.A.append(list(range(self.NumberOfColumns)))
            self.A[self.counter][self.INDEX] = self.counter
            self.A[self.counter][self.RJ] = ''
            self.A[self.counter][self.RJ1] = ''
            self.A[self.counter][self.QJ1] = ''
            self.A[self.counter][self.RJ2] = ''
            self.A[self.counter][self.SJ] = self.A[self.counter - 2][self.SJ] - \
                self.A[self.counter - 2][self.QJ1] * \
                self.A[self.counter - 1][self.SJ]
            self.A[self.counter][self.TJ] = self.A[self.counter - 2][self.TJ] - \
                self.A[self.counter - 2][self.QJ1] * \
                self.A[self.counter - 1][self.TJ]

            # Bezout’s coefficients
            self.s = self.A[self.counter][self.SJ]
            self.t = self.A[self.counter][self.TJ]

    # function to convert to subscript
    def get_sub(self, x):
        normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
        sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
        res = x.maketrans(''.join(normal), ''.join(sub_s))
        return x.translate(res)

    def printResults(self):
        print("GCD =", self.gcd)
        # GCD = s * a + t * b
        print("Bezout’s coefficients:")
        print(str(self.gcd) + " = " + str(self.s) + " * " + str(self.a) + " + " +
              str(self.t) + " * " + str(self.b))

        head = ['j',
                'r{}'.format(self.get_sub('j')),
                'r{}'.format(self.get_sub('j+1')),
                'q{}'.format(self.get_sub('j+1')),
                'r{}'.format(self.get_sub('j+2')),
                's{}'.format(self.get_sub('j')),
                't{}'.format(self.get_sub('j')), ]

        print(tabulate(self.A, headers = head, tablefmt = "grid"))


def main():
    print("Enter the number of the Algorithm:")
    print("1- Eratosthenes")
    print("2- Trial Division")
    print("3- Extended_Euclidean")
    print("4- Chinese remainder")
    print("5- Miller 's Test")

    number = abs(int(input()))
    if(number == 1):
        num = abs(int(input("Enter number: ")))
        eratosthenes = Eratosthenes(num)
        print(eratosthenes.integerNumbers)

    elif(number == 2):
        num = abs(int(input("Enter number: ")))
        trialDivision = TrialDivision(num)

    elif(number == 3):
        Extended_Euclidean()

    elif(number == 4):
        ChineseRemainder()

    elif(number == 5):
        Miller_Test()

    else:
        print("Error! Not supported!!")
        

main()
