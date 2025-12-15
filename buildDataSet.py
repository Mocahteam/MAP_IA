from abc import abstractmethod
from enum import Enum
from random import randint, random, choice
import argparse

# Ce script permet de construire le jeu de test. Le paramètre --rep_learn permet d'indiquer pour chaque scénario (boucle simple, boucle avec une option, deux boucles en séquence...) combien de paires seront générés, ce dataset a vocation à être utilisé pour l'apprentissage. Le paramètre --rep_test permet d'indiquer pour chaque scénario combien de paires seront générés, ce dataset a vocation à être utilisé pour les tests de performance. Le paramètre --len donne une limite sur la longueur de la séquence non compressée. Chaque fichier généré est donc un liste de paire sous la forme : séquence non compressée\tséquence compressée\n
# Ex :
# AAAAAAAA\t[A]\n
# TDTDTDTDTDTDTD\t[TD]\n
# ...

parser = argparse.ArgumentParser(description="Définir le nombre de répétitions de chaque cas.")
parser.add_argument("--rep_learn", type=int, default=1000, help="Nombre de répétitions de chaque scénario pour l'apprentissage (défaut: 1000)")
parser.add_argument("--rep_test", type=int, default=10, help="Nombre de répétitions de chaque scénario pour le test (défaut: 10)")
parser.add_argument("--len", type=int, default=50, help="Longueur maximale de la séquence (défaut: 50)")

args = parser.parse_args()

rep_learn = args.rep_learn
rep_test = args.rep_test

def getRandomChar(excludeList:list[str] = []) -> str:
    alphabet = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    filteredAlphabet = [c for c in alphabet if c not in excludeList]
    return choice(filteredAlphabet)

class Position(Enum):
    FIRST = 1
    MIDDLE = 2
    LAST = 3

class Builder:
    @abstractmethod
    def build(self, length:int) -> str:
        pass
    
    @abstractmethod
    def getCompression(self, withOpt = False) -> str:
        pass
    
    # retourne la longueur minimale d'une itération (donc sans option si ce builder peut en produire)
    @abstractmethod
    def getMinimalLength(self) -> int:
        pass

class TokenBuilder:
    def __init__(self, token:str):
        self.token:str = token

    # Quelque soit la longueur demandée on n'en place toujours qu'un
    def build(self, length:int = 1) -> str:
        return self.token
    
    def getCompression(self, withOpt = False) -> str:
        if withOpt and self.token != "":
            return ("*".join(self.token))+"*"
        else:
            return self.token
    
    # retourne la longueur minimale d'une itération (donc sans option si ce builder peut en produire)
    def getMinimalLength(self) -> int:
        return len(self.token)

class LoopBuilder(Builder):
    def __init__(self, token1:TokenBuilder, token2:TokenBuilder=TokenBuilder(""), frequency:float=0, optPosition:Position=Position.LAST, optBuilder:Builder=TokenBuilder("")):
        self.token1:TokenBuilder = token1
        self.token2:TokenBuilder = token2
        self.frequency:float = frequency
        self.optPosition:Position = optPosition
        self.optBuilder:Builder = optBuilder

        # Variables accumulant les données sur plusieurs appels de buil
        self.historyOpt:list[int] = []
        self.historyCptLoop:list[int] = []
        self.onlyOneOptAndIsAllwaysLast:bool = True
        self.onlyOneOptAndIsAllwaysFirst:bool = True
        
    def build(self, length:int) -> str:
        self.historyOpt.append(0)
        self.historyCptLoop.append(0)
        result:str = ""
        cptTokens:int = 0
        # Contrairement à self.cptLoop, cette variable local compte le nombre d'itération de ce build
        cptLoop:int = 0
        onlyOneOptAndIsFirst:bool = False
        onlyOneOptAndIsLast:bool = False
        while cptTokens <= length - self.getMinimalLength():
            self.historyCptLoop[-1] += 1
            cptLoop += 1
            # token1 et token2 sont forcément placés, donc on les comptabilise
            cptTokens += self.token1.getMinimalLength()+self.token2.getMinimalLength()

            if self.optPosition == Position.LAST:
                result += self.token1.build()+self.token2.build()
            elif self.optPosition == Position.MIDDLE:
                result += self.token1.build()

            # Si le tirage aléatoire dit qu'il faut en placer une et qu'il y a la place pour la mettre
            onlyOneOptAndIsLast = False
            if random() < self.frequency and length - cptTokens >= self.optBuilder.getMinimalLength():
                onlyOneOptAndIsFirst = False
                currentOptLength:int
                # Si on a la place de rentrer plus d'un pattern complet, prendre une longueur aléatoire
                if length-cptTokens > self.getMinimalLength():
                    currentOptLength = randint(self.optBuilder.getMinimalLength(), int((length-cptTokens)/(2 if cptLoop == 1 else 1))) # le /(2 if cptLoop == 1 else 1) permet si on est sur le premier tour de boucle de ne pas consommer toute la longueur et s'assurer qu'au moins une deuxième itération va pouvoir rentrer, donc au maximum on ne consomme que la moitié de la trace
                # Sinon, compléter sur ce qui reste
                else:
                    currentOptLength = length-cptTokens
                optResult = self.optBuilder.build(currentOptLength)
                result += optResult
                cptTokens += len(optResult)
                self.historyOpt[-1] += 1
                if self.optPosition == Position.FIRST:
                    onlyOneOptAndIsFirst = (cptLoop == 1)
                elif self.optPosition == Position.LAST:
                    onlyOneOptAndIsLast = (self.historyOpt[-1] == 1)
            
            if self.optPosition == Position.MIDDLE:
                result += self.token2.build()
            elif self.optPosition == Position.FIRST:
                result += self.token1.build()+self.token2.build()

        # C'est le premier appel si c'était le cas pour tous les builds précédent et que c'est aussi le cas pour celui là
        self.onlyOneOptAndIsAllwaysFirst = True if self.onlyOneOptAndIsAllwaysFirst and onlyOneOptAndIsFirst else False
        # C'est le dernier appel si c'était le cas pour tous les builds précédent et que c'est aussi le cas pour celui là
        self.onlyOneOptAndIsAllwaysLast = True if self.onlyOneOptAndIsAllwaysLast and onlyOneOptAndIsLast else False
        
        return result
    
    def getCompression(self, withOpt = False) -> str:
        # S'il n'y a toujours eu qu'une seule itération
        if all(x == 1 for x in self.historyCptLoop):
            # Pas d'option
            if max(self.historyOpt) == 0:
                return self.token1.getCompression(withOpt)+self.token2.getCompression(withOpt)
            # Si la seule option est toujours en premier
            elif self.onlyOneOptAndIsAllwaysFirst:
                return self.optBuilder.getCompression(withOpt)+self.token1.getCompression(withOpt)+self.token2.getCompression(withOpt)
            # Si la seule option est toujours en dernier
            elif self.onlyOneOptAndIsAllwaysLast:
                return self.token1.getCompression(withOpt)+self.token2.getCompression(withOpt)+self.optBuilder.getCompression(withOpt)
            # Sinon on a au moins une option au milieu
            else:
                # Vérifier si on a eu une option à chaque fois
                allwaysOpt:bool = sum(self.historyOpt) == sum(self.historyCptLoop)
                if self.optPosition == Position.FIRST:
                    return self.optBuilder.getCompression(not allwaysOpt)+self.token1.getCompression(withOpt)+self.token2.getCompression(withOpt)
                if self.optPosition == Position.MIDDLE:
                    return self.token1.getCompression(withOpt)+self.optBuilder.getCompression(not allwaysOpt)+self.token2.getCompression(withOpt)
                if self.optPosition == Position.LAST:
                    return self.token1.getCompression(withOpt)+self.token2.getCompression(withOpt)+self.optBuilder.getCompression(not allwaysOpt)
        else:
            # Pas d'option
            if max(self.historyOpt) == 0:
                return "["+self.token1.getCompression()+self.token2.getCompression()+"]"+("*" if withOpt else "")
            # Si la seule option est toujours en premier
            elif self.onlyOneOptAndIsAllwaysFirst:
                return self.optBuilder.getCompression(withOpt)+"["+self.token1.getCompression()+self.token2.getCompression()+"]"+("*" if withOpt else "")
            # Si la seule option est toujours en dernier
            elif self.onlyOneOptAndIsAllwaysLast:
                return "["+self.token1.getCompression()+self.token2.getCompression()+"]"+("*" if withOpt else "")+self.optBuilder.getCompression(withOpt)
            # Si une option est sortie à chaque iteration
            elif sum(self.historyOpt) == sum(self.historyCptLoop):
                if self.optPosition == Position.FIRST:
                    return "["+self.optBuilder.getCompression()+self.token1.getCompression()+self.token2.getCompression()+"]"+("*" if withOpt else "")
                elif self.optPosition == Position.MIDDLE:
                    return "["+self.token1.getCompression()+self.optBuilder.getCompression()+self.token2.getCompression()+"]"+("*" if withOpt else "")
                elif self.optPosition == Position.LAST:
                    return "["+self.token1.getCompression()+self.token2.getCompression()+self.optBuilder.getCompression()+"]"+("*" if withOpt else "")
            # Cas général on a quelques options de temps en temps
            else:
                if self.optPosition == Position.MIDDLE:
                    return "["+self.token1.getCompression()+self.optBuilder.getCompression(True)+self.token2.getCompression()+"]"+("*" if withOpt else "")
                else:
                    return "["+self.optBuilder.getCompression(True)+self.token1.getCompression()+self.token2.getCompression()+self.optBuilder.getCompression(True)+"]"+("*" if withOpt else "")
    
    # retourne la longueur minimale d'une itération
    def getMinimalLength(self) -> int:
        return self.token1.getMinimalLength()+self.token2.getMinimalLength()+(self.optBuilder.getMinimalLength() if self.frequency == 1 else 0)

# Nommenclature du nommage des différents cas : T <=> Token, B <=> Begin, E <=> End, o <=> optional

# Répétition de 1 caractère
# ex: [B]
def caseBTE(length):
    
    builder = LoopBuilder(TokenBuilder(getRandomChar()))
    result = builder.build(length)
    return result+"\t"+builder.getCompression()+"\n"

# Répétition de 1 caractère avec 1 au début et 1 a la fin
# ex: G[A]F
def caseTBTET(length):
    char1 = getRandomChar()
    char2 = getRandomChar([char1])
    builder = LoopBuilder(TokenBuilder(getRandomChar([char1, char2])))

    result = char1
    result += builder.build(length)
    result += char2
    
    return result+"\t"+char1+builder.getCompression()+char2+"\n"

# Répétition de 2 caractère avec 1 au début et 1 a la fin
# ex: R[TS]Y
def caseTBTTET(length):
    char1 = getRandomChar()
    char2 = getRandomChar([char1])
    innerChar1 = getRandomChar([char1, char2])
    innerChar2 = getRandomChar([char1, char2, innerChar1])
    builder = LoopBuilder(TokenBuilder(innerChar1+innerChar2))

    result = char1
    result += builder.build(length)
    result += char2
    
    return result+"\t"+char1+builder.getCompression()+char2+"\n"

# Répétition de 3 caractère avec 1 au début et 1 a la fin
# ex: K[GSP]Y
def caseTBTTTET(length):
    char1 = getRandomChar()
    char2 = getRandomChar([char1])
    innerChar1 = getRandomChar([char1, char2])
    innerChar2 = getRandomChar([char1, char2, innerChar1])
    innerChar3 = getRandomChar([char1, char2, innerChar1, innerChar2])
    builder = LoopBuilder(TokenBuilder(innerChar1+innerChar2+innerChar3))

    result = char1
    result += builder.build(length)
    result += char2
    
    return result+"\t"+char1+builder.getCompression()+char2+"\n"

# Répétition de 2 caractères avec X% de chance d'en avoir un supplémentaire au milieu
# ex: [IS*K]
def caseBTToTE(length, frequency):
    char1 = getRandomChar()
    char2 = getRandomChar([char1])
    optChar = getRandomChar([char1, char2])

    builder = LoopBuilder(TokenBuilder(char1), TokenBuilder(char2), frequency, Position.MIDDLE, TokenBuilder(optChar))

    return builder.build(length)+"\t"+builder.getCompression()+"\n"

# Répétition de 2 caractères avec X% de chance d'en avoir un supplémentaire aux extrémités
# ex: [J*ABJ*]
def caseBToTTToE(length, frequency):
    char1 = getRandomChar()
    char2 = getRandomChar([char1])
    optChar = getRandomChar([char1, char2])

    builder = LoopBuilder(TokenBuilder(char1), TokenBuilder(char2), frequency, Position.FIRST if random() < 0.5 else Position.LAST, TokenBuilder(optChar))

    return builder.build(length)+"\t"+builder.getCompression()+"\n"

# Répétition de 1 caractère 2 fois
# ex: [G][P]
def caseBTEBTE(length):
    char1 = getRandomChar()
    builder1 = LoopBuilder(TokenBuilder(char1))
    char2 = getRandomChar([char1])
    builder2 = LoopBuilder(TokenBuilder(char2))
    
    firstPart = randint(2,length-2)
    secondPart = randint(2,length-firstPart)
    
    result = builder1.build(firstPart)
    result += builder2.build(secondPart)
    
    return result+"\t"+builder1.getCompression()+builder2.getCompression()+"\n"

# Répétition de 1 caractère + Répétition 1 caractère
# ex: [Q[L]]
def caseBTBTEE(length):
    char = getRandomChar()
    innerChar = getRandomChar([char])
    innerBuilder = LoopBuilder(TokenBuilder(innerChar))

    builder = LoopBuilder(TokenBuilder(char), frequency=1, optPosition=Position.LAST, optBuilder=innerBuilder)

    return builder.build(length)+"\t"+builder.getCompression()+"\n"

# Répétition de répétition 1 caractère + suivi de 1 caractère
# ex: [[C]N]
def caseBBTETE(length):
    char = getRandomChar()
    innerChar = getRandomChar([char])
    innerBuilder = LoopBuilder(TokenBuilder(innerChar))

    builder = LoopBuilder(TokenBuilder(char), frequency=1, optPosition=Position.FIRST, optBuilder=innerBuilder)

    return builder.build(length)+"\t"+builder.getCompression()+"\n"

# Répétition de 1 caractère + Répétition 2 caractère
# ex: [V[DO]] 
def caseBTBTTEE(length):
    char = getRandomChar()
    innerChar1 = getRandomChar([char])
    innerChar2 = getRandomChar([char, innerChar1])
    innerBuilder = LoopBuilder(TokenBuilder(innerChar1+innerChar2))

    builder = LoopBuilder(TokenBuilder(char), frequency=1, optPosition=Position.LAST, optBuilder=innerBuilder)

    return builder.build(length)+"\t"+builder.getCompression()+"\n"

# Répétition de répétition 2 caractère + suivi de 1 caractère
# ex: [[FE]K]
def caseBBTTETE(length):
    char = getRandomChar()
    innerChar1 = getRandomChar([char])
    innerChar2 = getRandomChar([char, innerChar1])
    innerBuilder = LoopBuilder(TokenBuilder(innerChar1+innerChar2))

    builder = LoopBuilder(TokenBuilder(char), frequency=1, optPosition=Position.FIRST, optBuilder=innerBuilder)

    return builder.build(length)+"\t"+builder.getCompression()+"\n"

# X% de chance qu'un caractère soit présent à l'intérieur d'une boucle imbriquée
# ex: [A[BC*D]]
def caseBTBTToTEE(length, frequency):
    char = getRandomChar()
    innerChar1 = getRandomChar([char])
    innerChar2 = getRandomChar([char, innerChar1])
    innerCharOpt = getRandomChar([char, innerChar1, innerChar2])
    innerBuilder = LoopBuilder(TokenBuilder(innerChar1), TokenBuilder(innerChar2), frequency, Position.MIDDLE, TokenBuilder(innerCharOpt))

    builder = LoopBuilder(TokenBuilder(char), frequency=1, optPosition=Position.LAST, optBuilder=innerBuilder)

    return builder.build(length)+"\t"+builder.getCompression()+"\n"

# X% de chance d'avoir une boucle imbriquée répétant 2 caractères apparaissant aux extrémités
# ex: [[BC]*A[BC]*]
def caseBBTTEoTBTTEoE(length, frequency):
    char = getRandomChar()
    innerChar1 = getRandomChar([char])
    innerChar2 = getRandomChar([char, innerChar1])
    innerBuilder = LoopBuilder(TokenBuilder(innerChar1+innerChar2))

    builder = LoopBuilder(TokenBuilder(char), frequency=frequency, optPosition=Position.FIRST if random() < 0.5 else Position.LAST, optBuilder=innerBuilder)

    return builder.build(length)+"\t"+builder.getCompression()+"\n"

# X% de chance d'avoir un caractère aux extrémités d'une boucle imbriquée
# ex: [A[C*BC*]]
def caseBTBToTToEE(length, frequency):
    char = getRandomChar()
    innerChar = getRandomChar([char])
    innerCharOpt = getRandomChar([char, innerChar])
    innerBuilder = LoopBuilder(TokenBuilder(innerChar), frequency=frequency, optPosition=Position.FIRST if random() < 0.5 else Position.LAST, optBuilder=TokenBuilder(innerCharOpt))

    builder = LoopBuilder(TokenBuilder(char), frequency=1, optPosition=Position.LAST, optBuilder=innerBuilder)

    return builder.build(length)+"\t"+builder.getCompression()+"\n"

# Deux boucles en séquences séparées par au maximum 10 caractères
# ex: A[B]CHO[D]E
def caseTBTETTTBTET(length):
    usedChar = []
    char1 = getRandomChar()
    usedChar += [char1]
    char2 = getRandomChar(usedChar)
    usedChar += [char2]
    middle = ""
    char4 = getRandomChar(usedChar)
    usedChar += [char4]
    char5 = getRandomChar(usedChar)
    usedChar += [char5]

    loop1 = LoopBuilder(TokenBuilder(char2))
    loop2 = LoopBuilder(TokenBuilder(char4))

    # On retient 6 à la longueur pour pouvoir placer le début (char1), la fin (char5) et deux exemplaires de chaque boucle (2*char2+2*char4)
    # Dans cet exemple on ne souhaite au maximum que 10 tokens intercalés
    intercalatedTokkens = randint(2,min(length-6, 10))
    firstLoop = randint(2,length-intercalatedTokkens-4)
    secondLoop = randint(2,length-intercalatedTokkens-firstLoop-2)
    
    result:str = char1
    result += loop1.build(firstLoop)
    for _ in range(intercalatedTokkens):
        charX = getRandomChar(usedChar)
        middle += charX
        usedChar += [charX]
        result += charX
    result += loop2.build(secondLoop)
    result += char5

    return result+"\t"+char1+loop1.getCompression()+middle+loop2.getCompression()+char5+"\n"

# Une boucle suivie d'une boucle imbriquée
# ex: A[B][C[D]]
def caseTBTEBTBTEE(length):
    char1 = getRandomChar()
    char2 = getRandomChar([char1])
    char3 = getRandomChar([char1, char2])
    char4 = getRandomChar([char1, char2, char3])

    firstLoop = LoopBuilder(TokenBuilder(char2))
    innerLoop = LoopBuilder(TokenBuilder(char4))
    secondLoop = LoopBuilder(TokenBuilder(char3), frequency=1, optPosition=Position.LAST, optBuilder=innerLoop)

    # le 5 car il faut au minimum 5 token pour pouvoir construire [C[D]]
    # le 3 car on réserve au moins trois tokens pour construire A[B]
    secondLoopLength = randint(5, length-3)
    firstLoopIter = randint(2, length-secondLoopLength-1)

    result = char1
    result += firstLoop.build(firstLoopIter)
    result += secondLoop.build(secondLoopLength)

    return result+"\t"+char1+firstLoop.getCompression()+secondLoop.getCompression()+"\n"

def getEssence(str:str):
    return ''.join(["_" if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" else c for c in str if c in str])

def writeSet(fichier, repetition, length, caseFunc, minlength, essence, frequency=None):
    cpt = 0
    while cpt < repetition:
        if frequency != None:
            result = caseFunc(randint(minlength,length), frequency)
        else:
            result = caseFunc(randint(minlength,length))
        if getEssence(result.split("\t")[1][:-1]) == essence:
            fichier.write(result)
            cpt += 1

def writeDataSet(fichier, repetition, length):
    # 1. Répétition de 1 caractère
    # ex: [B]
    writeSet(fichier, repetition, length, caseBTE, 2, "[_]")
    
    # 2. Répétition de 1 caractère avec 1 au début et 1 a la fin
    # ex: G[A]F
    writeSet(fichier, repetition, length, caseTBTET, 4, "_[_]_")
    
    # 3. Répétition de 2 caractère avec 1 au début et 1 a la fin
    # ex: R[TS]Y
    writeSet(fichier, repetition, length, caseTBTTET, 6, "_[__]_")
    
    # 4. Répétition de 3 caractère avec 1 au début et 1 a la fin
    # ex: K[GSP]Y
    writeSet(fichier, repetition, length, caseTBTTTET, 8, "_[___]_")
    
    # 5. Répétition de 1 caractère 2 fois
    # ex: [G][P]
    writeSet(fichier, repetition, length, caseBTEBTE, 4, "[_][_]")
    
    # 6. Répétition de 1 caractère + Répétition 1 caractère
    # ex: [Q[L]]
    writeSet(fichier, repetition, length, caseBTBTEE, 5, "[_[_]]")
    
    # 7. Répétition de répétition 1 caractère + suivi de 1 caractère
    # ex: [[C]N]
    writeSet(fichier, repetition, length, caseBBTETE, 5, "[[_]_]")
    
    # 8. Répétition de 1 caractère + Répétition 2 caractère
    # ex: [V[DO]] 
    writeSet(fichier, repetition, length, caseBTBTTEE, 8, "[_[__]]")
    
    # 9. Répétition de répétition 2 caractère + suivi de 1 caractère
    # ex: [[FE]K]
    writeSet(fichier, repetition, length, caseBBTTETE, 8, "[[__]_]")
    
    # 10. Deux boucles en séquences séparées par au maximum 10 caractères
    # ex: A[B]CHO[D]E
    # Pas de vérification pour ce cas là mais on ne devrait pas avoir autre chose que ce qui est attendu
    for _ in range(repetition):
        fichier.write(caseTBTETTTBTET(randint(8,length)))
    
    # 11. Une boucle suivie d'une boucle imbriquée
    # ex: A[B][C[D]]
    writeSet(fichier, repetition, length, caseTBTEBTBTEE, 8, "_[_][_[_]]")
    
    # 12.1. Répétition de 2 caractères avec 10% de chance d'en avoir un supplémentaire au milieu
    # ex: [IS*K]
    writeSet(fichier, repetition/2, length, caseBTToTE, 5, "[__*_]", 0.1)
    
    # 12.2. Répétition de 2 caractères avec 90% de chance d'en avoir un supplémentaire au milieu
    # ex: [KD*P]
    writeSet(fichier, repetition/2, length, caseBTToTE, 5, "[__*_]", 0.9)
    
    # 13.1. Répétition de 2 caractères avec 10% de chance d'en avoir un supplémentaire aux extrémités
    # ex: [J*ABJ*]
    writeSet(fichier, repetition/2, length, caseBToTTToE, 5, "[_*___*]", 0.1)
    
    # 13.2. Répétition de 2 caractères avec 90% de chance d'en avoir un supplémentaire aux extrémités
    # ex: [J*ABJ*]
    writeSet(fichier, repetition/2, length, caseBToTTToE, 5, "[_*___*]", 0.9)
    
    # 14.1. 10% de chance d'avoir une boucle imbriquée répétant 2 caractères apparaissant aux extrémités
    # ex: [[BC]*A[BC]*]
    writeSet(fichier, repetition/2, length, caseBBTTEoTBTTEoE, 6, "[[__]*_[__]*]", 0.1)
    
    # 14.2. 90% de chance d'avoir une boucle imbriquée répétant 2 caractères apparaissant aux extrémités
    # ex: [[BC]*A[BC]*]
    writeSet(fichier, repetition/2, length, caseBBTTEoTBTTEoE, 6, "[[__]*_[__]*]", 0.9)
    
    # 15.1. 10% de chance qu'un caractère soit présent à l'intérieur d'une boucle imbriquée
    # ex: [A[BC*D]]
    writeSet(fichier, repetition/2, length, caseBTBTToTEE, 9, "[_[__*_]]", 0.1)
    
    # 15.2. 90% de chance qu'un caractère soit présent à l'intérieur d'une boucle imbriquée
    # ex: [A[BC*D]]
    writeSet(fichier, repetition/2, length, caseBTBTToTEE, 9, "[_[__*_]]", 0.9)
    
    # 16.1. 10% de chance d'avoir un caractère aux extrémités d'une boucle imbriquée
    # ex: [A[C*BC*]]
    writeSet(fichier, repetition/2, length, caseBTBToTToEE, 6, "[_[_*__*]]", 0.1)
    
    # 16.2. 90% de chance d'avoir un caractère aux extrémités d'une boucle imbriquée
    # ex: [A[C*BC*]]
    writeSet(fichier, repetition/2, length, caseBTBToTToEE, 6, "[_[_*__*]]", 0.9)
    
    fichier.close()
    

# Génération des deux dataset
writeDataSet(open("long-comp_forlearning.txt", "w"), rep_learn, args.len)
writeDataSet(open("long-comp_fortesting.txt", "w"), rep_test, args.len)