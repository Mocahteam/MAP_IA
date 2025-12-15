from abc import abstractmethod
from enum import Enum
from random import randint, random, choice
import argparse

# Ce script permet de construire un jeu de test proche du dataset d'entrainement mais légèrement divergent. Le paramètre --nb_rep permet d'indiquer pour chaque scénario combien de paires seront générés. Le paramètre --len donne une limite sur la longueur de la séquence non compressée. Le résultat est donc un liste de paire sous la forme : séquence non compressée\tséquence compressée\n
# Ex :
# AAAAAAAA\t[A]\n
# TDTDTDTDTDTDTD\t[TD]\n
# ...

parser = argparse.ArgumentParser(description="Définir le nombre de répétitions de chaque cas.")
parser.add_argument("--nb_rep", type=int, default=10, help="Nombre de répétitions de chaque scénario pour le test (défaut: 10)")
parser.add_argument("--len", type=int, default=50, help="Longueur maximale de la séquence (défaut: 50)")

args = parser.parse_args()

nb_rep = args.nb_rep

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

        # Variables accumulant les données sur plusieurs appels de build
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

# Répétition de 4 à 6 caractères
# ex: [BTYS]
def caseBTxE(length, innerLength):
    usedChar = []
    innerTokens = ""
    for _ in range(innerLength):
        charX = getRandomChar(usedChar)
        innerTokens += charX
        usedChar += [charX]
    builder = LoopBuilder(TokenBuilder(innerTokens))
    result = builder.build(length)
    return result+"\t"+builder.getCompression()+"\n"

# Répétition de 1 caractère avec n au début et m a la fin
# ex: GD[A]FJU
def caseTxBTETx(length, beginlength, endLength):
    usedChar = []
    begin = ""
    for _ in range(beginlength):
        charX = getRandomChar(usedChar)
        begin += charX
        usedChar += [charX]

    end = ""
    for _ in range(endLength):
        charX = getRandomChar(usedChar)
        end += charX
        usedChar += [charX]

    builder = LoopBuilder(TokenBuilder(getRandomChar(usedChar)))
    middle = builder.build(length-beginlength-endLength)
    
    return begin+middle+end+"\t"+begin+builder.getCompression()+end+"\n"

# ex: [AB*C*D]
def caseBTToToTE(length, frequency):
    char1 = getRandomChar()
    char2 = getRandomChar([char1])
    optChar1 = getRandomChar([char1, char2])
    optChar2 = getRandomChar([char1, char2, optChar1])

    builder = LoopBuilder(TokenBuilder(char1), TokenBuilder(char2), frequency, Position.MIDDLE, TokenBuilder(optChar1+optChar2))

    return builder.build(length)+"\t"+builder.getCompression()+"\n"

# ex: [A*BA*]
def caseBToTToE(length, frequency):
    char1 = getRandomChar()
    optChar = getRandomChar([char1])

    builder = LoopBuilder(token1=TokenBuilder(char1), frequency=frequency, optPosition=Position.FIRST if random() < 0.5 else Position.LAST, optBuilder=TokenBuilder(optChar))

    return builder.build(length)+"\t"+builder.getCompression()+"\n"

# ex: [G][P][T]
def caseBTEBTEBTE(length):
    char1 = getRandomChar()
    builder1 = LoopBuilder(TokenBuilder(char1))
    char2 = getRandomChar([char1])
    builder2 = LoopBuilder(TokenBuilder(char2))
    char3 = getRandomChar([char1, char2])
    builder3 = LoopBuilder(TokenBuilder(char3))
    
    firstPart = randint(2,length-4)
    secondPart = randint(2,length-firstPart-2)
    thirdPart = randint(2,length-firstPart-secondPart)
    
    result = builder1.build(firstPart)
    result += builder2.build(secondPart)
    result += builder3.build(thirdPart)
    
    return result+"\t"+builder1.getCompression()+builder2.getCompression()+builder3.getCompression()+"\n"

# ex: [GP][T]
def caseBTTEBTE(length):
    char1 = getRandomChar()
    char2 = getRandomChar([char1])
    builder1 = LoopBuilder(TokenBuilder(char1+char2))
    char3 = getRandomChar([char1, char2])
    builder2 = LoopBuilder(TokenBuilder(char3))
    
    firstPart = randint(2,length-2)
    secondPart = randint(2,length-firstPart)
    
    result = builder1.build(firstPart)
    result += builder2.build(secondPart)
    
    return result+"\t"+builder1.getCompression()+builder2.getCompression()+"\n"

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
    # ex: [BTYS] => répétition de 2 à 6 caractères
    # Pas de vérification pour ce cas là mais on ne devrait pas avoir autre chose que ce qui est attendu
    for _ in range(repetition):
        fichier.write(caseBTxE(randint(12,length), randint(2,6)))

    # ex: GD[A]FJU
    # Pas de vérification pour ce cas là mais on ne devrait pas avoir autre chose que ce qui est attendu
    for _ in range(repetition):
        fichier.write(caseTxBTETx(randint(10,length), randint(2,4), randint(2,4)))
    
    # ex: [AB*C*D]
    writeSet(fichier, repetition/2, length, caseBTToToTE, 6, "[__*_*_]", 0.1)
    writeSet(fichier, repetition/2, length, caseBTToToTE, 6, "[__*_*_]", 0.9)

    # ex: [A*BA*]
    writeSet(fichier, repetition/2, length, caseBToTToE, 3, "[_*__*]", 0.1)
    writeSet(fichier, repetition/2, length, caseBToTToE, 3, "[_*__*]", 0.9)
    
    # ex: [G][P][T]
    writeSet(fichier, repetition, length, caseBTEBTEBTE, 6, "[_][_][_]")

    # ex: [GP][T]
    writeSet(fichier, repetition, length, caseBTTEBTE, 6, "[__][_]")
    
    fichier.close()
    

# Génération du dataset
writeDataSet(open("long-comp_alternative.txt", "w"), nb_rep, args.len)