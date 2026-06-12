from abc import abstractmethod
from enum import Enum
from random import randint, random, choice
import argparse

# Ce script permet de construire un dataset. Le paramètre --nb_rep permet d'indiquer pour chaque scénario combien de paires seront générés. Le paramètre --len donne une limite sur la longueur de la séquence non compressée. Le résultat est donc un liste de paire sous la forme : séquence non compressée\tséquence compressée\n. Le paramètre --output indique le nom du fichier dans lequel le résultat est écrit
# Ex :
# AAAAAAAA\t[A]\n
# TDTDTDTDTDTDTD\t[TD]\n
# ...

parser = argparse.ArgumentParser(description="Définir le nombre de répétitions de chaque cas.")
parser.add_argument("--nb_rep", type=int, default=10, help="Nombre de répétitions de chaque scénario pour le test (défaut: 10)")
parser.add_argument("--len", type=int, default=50, help="Longueur maximale de la séquence (défaut: 50)")
parser.add_argument("--output", type=str, default="output.txt", help="Nom du fichier de sortie (défaut: output.txt)")

args = parser.parse_args()

class Builder:
    def __init__(self, length:int, frequency:float = 1):
        self.length:int = length
        self.frequency:float = frequency
        self.allwaysPlaced:bool = True
        self.neverPlaced:bool = True
        self.firstPlaced:bool = False
        self.lastPlaced:bool = False
        self.loopCpt:int = 0
        self.placedCpt:int = 0
        
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def build(self) -> str:
        if self.frequency == 1 and self.length - self.getMaximalLength() < 0:
            raise Exception("longueur insuffisante")
        self.lastPlaced = self.frequency == 1 or random() < self.frequency
        if self.lastPlaced:
            self.placedCpt += 1
        self.allwaysPlaced = self.allwaysPlaced and self.lastPlaced
        self.neverPlaced = self.neverPlaced and not self.lastPlaced
        if self.loopCpt == 0 and self.lastPlaced:
            self.firstPlaced = True
        self.loopCpt += 1
        pass
    
    @abstractmethod
    def getLinear(self) -> list["Builder"]:
        pass
    
    @abstractmethod
    def getCompression(self) -> str:
        pass
    
    # retourne la longueur minimale d'une itération (donc sans option si ce builder peut en produire)
    @abstractmethod
    def getMinimalLength(self) -> int:
        pass
    
    # retourne la longueur maximale d'une itération (donc avec option si ce builder peut en produire)
    @abstractmethod
    def getMaximalLength(self) -> int:
        pass

    def export(self) -> str:
        return self.build()+"\t"+self.getCompression()+"\n"

    def __str__(self):
        return "(len: "+str(self.length)+", freq: "+str(self.frequency)+", allways: "+str(self.allwaysPlaced)+", never: "+str(self.neverPlaced)+", first: "+str(self.firstPlaced)+", last: "+str(self.lastPlaced)+", loopCpt: "+str(self.loopCpt)+", placedCpt: "+str(self.placedCpt)+")"
    
class SeqBuilder(Builder):
    def __init__(self, part1:Builder, part2:Builder):
        super().__init__(part1.length+part2.length, 1)
        self.part1:Builder = part1
        self.part2:Builder = part2

    def __str__(self):
        return str(self.part1)+str(self.part2)+" "+super().__str__()
    
    def build(self) -> str:
        super().build()
        return self.part1.build()+self.part2.build()
    
    def getLinear(self) -> list["Builder"]:
        return self.part1.getLinear()+self.part2.getLinear()
    
    def getCompression(self) -> str:
        return self.part1.getCompression()+self.part2.getCompression()
    
    def getMinimalLength(self) -> int:
        return self.part1.getMinimalLength()+self.part2.getMinimalLength()
    
    def getMaximalLength(self) -> int:
        return self.part1.getMaximalLength()+self.part2.getMaximalLength()

class TokenBuilder(Builder):
    def __init__(self, token:str, frequency:float = 1):
        super().__init__(len(token), frequency)
        self.token:str = token

    def __str__(self):
        return self.token+" "+super().__str__()

    def build(self) -> str:
        super().build()
        return self.token if self.lastPlaced else ""
    
    def getLinear(self) -> list["Builder"]:
        return [self]
    
    def getCompression(self) -> str:
        if self.allwaysPlaced:
            return self.token
        elif self.neverPlaced:
            return ""
        else:
            return ("*".join(self.token))+"*"
    
    def getMinimalLength(self) -> int:
        return len(self.token) if self.frequency == 1 else 0
    
    def getMaximalLength(self) -> int:
        return len(self.token)

class LoopBuilder(Builder):
    def __init__(self, inner:Builder, length:int, frequency:float=1):
        super().__init__(length, frequency)
        self.inner:Builder = inner

    def __str__(self):
        return "["+str(self.inner)+"] "+super().__str__()
        
    def build(self) -> str:
        super().build()
        result:str = ""
        if self.lastPlaced:
            cptTokens:int = 0
            iterLength:int = randint(self.getMaximalLength()*(2 if self.loopCpt == 1 else 1), self.length) # Pour faire que chaque build ne produise pas toujours le même nombre d'occurence, on s'assure quand même que sur le premier on ait toujours au minimum deux itérations
            while cptTokens <= iterLength - self.getMaximalLength():
                innerResult:str = self.inner.build()
                result += innerResult
                cptTokens += len(innerResult)
        return result    
    
    def getLinear(self) -> list["Builder"]:
        return [self]
    
    def getCompression(self) -> str:
        if self.neverPlaced or self.placedCpt == 0:
            return ""
        else:
            content:list[Builder] = self.inner.getLinear()
            
            # extraire les évènements en début de boucle qui ne sont apparus qu'une fois et la première fois
            beginOnce:list[Builder] = []
            body:list[Builder]=[]
            for c in content:
                if c.firstPlaced and c.placedCpt == 1 and body == []:
                    c.allwaysPlaced = True # désactiver l'optionalité
                    beginOnce.append(c)
                else:
                    body.append(c)
            # extraire les évènements en fin de boucle qui ne sont apparus qu'une fois et la première fois
            endOnce:list[Builder] = []
            content=body
            body = []
            for c in reversed(content):
                if c.lastPlaced and c.placedCpt == 1 and body == []:
                    c.allwaysPlaced = True # désactiver l'optionalité
                    endOnce.append(c)
                else:
                    body.append(c)
            endOnce = list(reversed(endOnce))
            content = list(reversed(body))

            firstNonOption = next((i for i, x in enumerate(content) if x.allwaysPlaced and not x.neverPlaced), -1)
            lastNonOption = next((len(content)-1-i for i, x in enumerate(reversed(content)) if x.allwaysPlaced and not x.neverPlaced), -1)
            newContent = []
            if firstNonOption != -1 and lastNonOption != -1:
                newContent = content[lastNonOption+1:]+content[0:firstNonOption]+content[firstNonOption:lastNonOption+1]+content[lastNonOption+1:]+content[0:firstNonOption]
            comp = ""
            for builder in beginOnce:
                comp += builder.getCompression()
            comp += "[" if self.inner.placedCpt > 1 else ""
            for builder in newContent:
                comp += builder.getCompression()
            comp += "]" if self.inner.placedCpt > 1 else ""
            if not self.allwaysPlaced:
                comp += "*"
            for builder in endOnce:
                comp += builder.getCompression()
            #print (str(self))

            return comp
    
    def getMinimalLength(self) -> int:
        return self.inner.getMinimalLength() if self.frequency == 1 else 0
    
    def getMaximalLength(self) -> int:
        return self.inner.getMaximalLength()

def listToSeq(listBuilder:list[Builder]) -> Builder:
    if len(listBuilder) == 0:
        raise Exception("La liste ne doit pas être vide")
    elif len(listBuilder) == 1:
        return listBuilder[0]
    else:
        seq1:SeqBuilder = SeqBuilder(listBuilder[0], listBuilder[1])
        for b in listBuilder[2:]:
            seq2:SeqBuilder = SeqBuilder(seq1, b)
            seq1 = seq2
        return seq1

def getRandomChar(excludeList:list[str] = []) -> str:
    alphabet = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    filteredAlphabet = [c for c in alphabet if c not in excludeList]
    if len(filteredAlphabet) == 0:
        raise Exception("Tout le vocabulaire a été utilisé")
    else:
        return choice(filteredAlphabet)

# Créer une séquence de tokens d'une longueur comprise entre minLength et maxLength, si on veut une option il faut que minLength soit > 1
def tokenSeqBuilder(minLength:int, maxLength:int, usedChar:list[str], withOpt:bool = False) -> Builder:
    if minLength == 1 and withOpt:
        raise Exception("Avec WithOpt=True, minLength doit être > 1")
    innerLength:int = randint(minLength,maxLength)
    # Détermine le nombre d'option de cette séquence
    # 0 si on ne veut pas d'option ou que la longueur effective de la séquence est 1
    # >= 1 et < innerLength
    nbOpt = randint(1, innerLength-1) if withOpt else 0
    # déterminer la position des options
    listPos:list[int] = [i for i in range(innerLength)]
    optPos:list[int] = []
    for c in range(nbOpt):
        optPos.append(choice(listPos))
        listPos.remove(optPos[-1])
    #Construction de la séquence de tokens
    tokens:list[TokenBuilder] = []
    for cpt in range(innerLength):
        charX:str = getRandomChar(usedChar)
        frequency:float = 1 if cpt not in optPos else (0.1+random()*0.8) # tirer un nombre entre 0.1 et 0.9
        tok:TokenBuilder = TokenBuilder(charX, frequency)
        tokens += [tok]
        usedChar += [charX]
    return listToSeq(tokens)

# Boucle sans option ayant des trucs avant et/ou après
# ex: AB[LSTM]DEF
def LoopWithContextBuilder(length:int, usedChar:list[str], withInnerOpt:bool=False, withContextOpt:bool=False, withLoopOpt:bool=False) -> Builder:
    config:int = randint(1, 3) if not withLoopOpt else 3 # 1 <=> onlyBegin, 2 <=> onlyEnd, 3 <=> both. Dans le cas où on veut des loop optionnelles on se force en mode 3 pour avoir des choses avant et après (cas plus simple mais c'est déjà pas mal)
    listBuilder:list[Builder] = []
    # Si withContextOpt on force la minimum à 2 au lieu de 1 pour être sûr qu'au moins un élément de contexte sera affiché
    # Si withLoopOpt on force le maximum à 2 pour réduire le contexte et laisser plus de place à la boucle optionnelle
    beginLength:int = randint(2 if withContextOpt else 1, 2 if withLoopOpt else 4) if config != 2 else 0
    endLength:int = randint(2 if withContextOpt else 1, 2 if withLoopOpt else 4) if config != 1 else 0
    if beginLength > 0:
        listBuilder.append(tokenSeqBuilder(beginLength, beginLength, usedChar, withContextOpt))
    innerBuilder:Builder = tokenSeqBuilder(2 if withInnerOpt else 1, 2 if withLoopOpt else 4, usedChar, withInnerOpt)
    listBuilder.append(LoopBuilder(innerBuilder, randint(innerBuilder.getMaximalLength()*2, length-beginLength-endLength), 1 if not withLoopOpt else (0.1+random()*0.8)))
    if endLength > 0:
        listBuilder.append(tokenSeqBuilder(endLength, endLength, usedChar, withContextOpt))
    return listToSeq(listBuilder)

# Deux boucles en séquence pouvant avoir un préfixe, un intercalaire et un suffixe
# ex: AB[C]DEF[G]H
# ex: [AB*C]D[E*FE*]G
def SequentialLoopBuilder(length:int, usedChar:list[str], withOpt:bool=False) -> Builder:
    config:int = randint(0, 7) # 0 <=> nowhere, 1 <=> onlyBegin, 2 <=> onlyMiddle, 3 <=> begin and middle, 4 <=> onlyEnd, 5 <=> begin and end, 6 <=> middle and end, 7 <=> everywhere
    listBuilder:list[Builder] = []
    beginLength:int = randint(1,4) if config & 1 == 1 else 0
    middleLength:int = randint(1,4) if config & 2 == 2 else 0
    endLength:int = randint(1,4) if config & 4 == 4 else 0
    innerBuilder1:Builder = tokenSeqBuilder(2 if withOpt else 1, 4, usedChar, withOpt)
    innerBuilder2:Builder = tokenSeqBuilder(2 if withOpt else 1, 4, usedChar, withOpt)
    inner1Length:int = randint(innerBuilder1.getMaximalLength()*2, length-innerBuilder2.getMaximalLength()*2-beginLength-middleLength-endLength)
    inner2Length:int = randint(innerBuilder2.getMaximalLength()*2, length-inner1Length-beginLength-middleLength-endLength)
    # add begin
    if config & 1 == 1:
        listBuilder.append(tokenSeqBuilder(beginLength, beginLength, usedChar))
    #add first loop
    listBuilder.append(LoopBuilder(innerBuilder1, inner1Length))
    # add middle
    if config & 2 == 2:
        listBuilder.append(tokenSeqBuilder(middleLength, middleLength, usedChar))
    # add second loop
    listBuilder.append(LoopBuilder(innerBuilder2, inner2Length))
    # add end
    if config & 4 == 4:
        listBuilder.append(tokenSeqBuilder(endLength, endLength, usedChar))
    return listToSeq(listBuilder)

# Deux boucles imbriquées pouvant avoir des préfixes et des suffixes
# ex: AB[CD[EF]G]H
# ex: AB[CD*[E*FE*]G]H
# ex: AB[C[DE]*F]G
def InnerLoopBuilder(length:int, usedChar:list[str], withTokenOpt:bool=False, withLoopOpt=False) -> Builder:
    config:int = randint(1, 3) # 1 <=> onlyBegin, 2 <=> onlyEnd, 3 <=> both
    listBuilder:list[Builder] = []
    beginLength:int = randint(1,4) if config != 2 else 0
    endLength:int = randint(1,4) if config != 1 else 0
    # Au pire des cas il faut réserver 16 tokens pour la boucle impriquée => 4 begin + 4 end + 4*2 body. Et comme il faut qu'elle tourne au moins une deuxième fois il ne faut pas quelle soit plus longue que la longueur restant à exploiter
    innerLoopLength:int = randint(16, (length-beginLength-endLength)//2)
    
    # Ajout du begin
    if config != 2:
        listBuilder.append(tokenSeqBuilder(beginLength, beginLength, usedChar))
    
    # Ajout de la boucle
    # Création du body du corps de la boucle => la boucle imbriquée
    innerBuilder:Builder = LoopWithContextBuilder(innerLoopLength, usedChar, withTokenOpt, withTokenOpt, withLoopOpt)
    # Comme le corps de la boucle doit au moins tourner deux fois il faut getMaximalLength()*2
    listBuilder.append(LoopBuilder(innerBuilder, randint(innerBuilder.getMaximalLength()*2, length-beginLength-endLength)))

    # Ajout du end
    if config != 1:
        listBuilder.append(tokenSeqBuilder(endLength, endLength, usedChar))
    return listToSeq(listBuilder)

def writeDataSet(fichier, repetition:int, length:int):
    
    # ex: LSTM => pas de compression possible
    for _ in range(repetition):
        builder:Builder = tokenSeqBuilder(2, 26, [])
        fichier.write(builder.export())
        
    # ex: [LSTM] => répétition de 1 à 4 caractères sans options
    for _ in range(repetition):
        usedChar:list[str] = []
        innerBuilder:Builder = tokenSeqBuilder(1, 4, usedChar)
        loopBuilder:LoopBuilder = LoopBuilder(innerBuilder, randint(innerBuilder.getMaximalLength()*2, length))
        fichier.write(loopBuilder.export())
    
    # ex: [LS*T*M] => répétition de 2 à 4 caractères avec options
    # ex: [L*S*TML*S*] => répétition de 2 à 4 caractères avec options
    for _ in range(repetition):
        usedChar:list[str] = []
        innerBuilder:Builder = tokenSeqBuilder(2, 4, usedChar, True)
        loopBuilder:LoopBuilder = LoopBuilder(innerBuilder, randint(innerBuilder.getMaximalLength()*2, length))
        fichier.write(loopBuilder.export())
    
    # Boucle sans option ayant des trucs avant et/ou après
    # ex: AB[LSTM]C
    for _ in range(repetition):
        fichier.write(LoopWithContextBuilder(length, []).export())
    
    # Boucle avec options ayant des trucs avant et/ou après
    # ex: AB[L*STML*]C
    for _ in range(repetition):
        fichier.write(LoopWithContextBuilder(length, [], True).export())
    
    # Boucles en séquence sans option avec éventuellement des trucs autour
    # ex: AB[LSTM]C[GPT]DEF
    # ex: [LSTM][GPT]
    for _ in range(repetition):
        fichier.write(SequentialLoopBuilder(length, []).export())
    
    # Boucles en séquence avec options et avec éventuellement des trucs autour
    # ex: AB[L*STML*]C[GPT]DEF
    # ex: [LST*M][GP*T]
    for _ in range(repetition):
        fichier.write(SequentialLoopBuilder(length, [], True).export())
    
    # Boucles imbriquées sans options
    for _ in range(repetition):
        fichier.write(InnerLoopBuilder(length, [], False, False).export())
    
    # Boucles imbriquées avec options
    for _ in range(repetition):
        fichier.write(InnerLoopBuilder(length, [], True, False).export())
        
    # Boucles imbriquées avec options
    for _ in range(repetition):
        fichier.write(InnerLoopBuilder(length, [], False, True).export())
    
    
    #test:list[Builder] = [LoopBuilder(LoopBuilder(SeqBuilder(TokenBuilder("R"), TokenBuilder("H", 0.3)), 10), 10)]
    #fichier.write(listToSeq(test).export())
    
    fichier.close()
    

# Génération du dataset
writeDataSet(open(args.output, "w"), args.nb_rep, args.len)