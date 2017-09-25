import numpy as np
#import json
#assumes the input is a matrix to the power of 2
#Forrest Dodds

#imports the matrix from a file
def importMatrix(fname):
    matrix = []
    try:
        file = open(fname, "r")
        linecount = 0
        for line in file:
            #print(line)
            #reads in each line, strips whitespace, and stores into an array
            line = line.rstrip().split(",")

            #makes the values of the line integers
            count = 0
            for x in line:
                #print("Value of x: " + x)
                line[count] = int(x)
                count += 1

            # print("Line: " + str(line) + " appended")
            matrix.append(line)
            linecount += 1
        file.close()
        npmatrix = np.array(matrix)
        return npmatrix
    except IOError:
        print("Error reading file.")
        return 0

#takes input of m1 (matrix 1) and m2 (matrix 2) and returns the product via classical matrix multiplication
def classicMatMult(m1,m2):
    # print("Classical Matrix Multiplication goes here.")

    # print ("Len of m1: " + str(len(m1)))
    #Base case
    if (len(m1) == len(m2) == 2):
        # print("Base case!")
        matrix = np.array([
            # [ AE + BG ] , [ AF + BH]
            [m1[0][0] * m2[0][0] + m1[0][1] * m2[1][0], m1[0][0] * m2[0][1] + m1[0][1] * m2[1][1]],
            # [ CE + DG ] , [ CF + DH]
            [m1[1][0] * m2[0][0] + m1[1][1] * m2[1][0], m1[1][0] * m2[0][1] + m1[1][1] * m2[1][1]]
        ])
        # print(matrix)
        return matrix
    elif(len(m1) == len(m2)):
        # Does matrix multiplication
        # m1 = [ [A, B], [C, D] ]
        ########################
        # m2 = [ [E, F], [G, H] ]
        # print(m1)
        # print(m2)

        splitSize = int(len(m1)/2)
        # print(splitSize)
        #slices the arrays more easily than classicMatMult
        A, B, C, D = m1[:splitSize, :splitSize], m1[:splitSize, splitSize:], m1[splitSize:, :splitSize], m1[splitSize:, splitSize:]
        E, F, G, H = m2[:splitSize, :splitSize], m2[:splitSize, splitSize:], m2[splitSize:, :splitSize], m2[splitSize:, splitSize:]
        # print(A,B,C,D)
        # print(E,F,G,H)

        AE = classicMatMult(A,E)
        BG = classicMatMult(B,G)
        AF = classicMatMult(A,F)
        BH = classicMatMult(B,H)
        CE = classicMatMult(C,E)
        DG = classicMatMult(D,G)
        CF = classicMatMult(C,F)
        DH = classicMatMult(D,H)

        # does the additions for the matrix
        AEpBG = np.add(AE,BG)
        # print("AEpBG: \n" + str(AEpBG))
        AFpBH = np.add(AF,BH)
        # print("AFpBH: \n" + str(AFpBH))
        CEpDG = np.add(CE,DG)
        # print("CEpDG: \n" + str(CEpDG))
        CFpDH = np.add(CF,DH)
        # print("CFpDH: \n" + str(CFpDH))

        #rebuilds the matrix
        matrix = np.array(np.concatenate([np.concatenate([AEpBG,AFpBH],axis=1),np.concatenate([CEpDG,CFpDH],axis=1)]))
        return matrix
    else:
        print("Something went wrong, sorry.")

#takes input of m1 (matrix 1) and m2 (matrix 2) and returns the product via Strassen matrix multiplication
def strassenMatMult(m1,m2):
    if (len(m1) == len(m2) == 2):
        # print("Base case!")
        matrix = np.array([
            # [ AE + BG ] , [ AF + BH]
            [m1[0][0] * m2[0][0] + m1[0][1] * m2[1][0], m1[0][0] * m2[0][1] + m1[0][1] * m2[1][1]],
            # [ CE + DG ] , [ CF + DH]
            [m1[1][0] * m2[0][0] + m1[1][1] * m2[1][0], m1[1][0] * m2[0][1] + m1[1][1] * m2[1][1]]
        ])
        # print(matrix)
        return matrix
    elif (len(m1) == len(m2)):
        splitSize = int(len(m1)/2)
        # print(splitSize)
        #slices the arrays more easily than classicMatMult
        A, B, C, D = m1[:splitSize, :splitSize], m1[:splitSize, splitSize:], m1[splitSize:, :splitSize], m1[splitSize:, splitSize:]
        E, F, G, H = m2[:splitSize, :splitSize], m2[:splitSize, splitSize:], m2[splitSize:, :splitSize], m2[splitSize:, splitSize:]
        # print(A,B,C,D)
        # print(E,F,G,H)

        P1 = strassenMatMult(A,np.subtract(F,H))
        P2 = strassenMatMult(np.add(A,B),H)
        P3 = strassenMatMult(np.add(C,D),E)
        P4 = strassenMatMult(D,np.subtract(G,E))
        P5 = strassenMatMult(np.add(A,D),np.add(E,H))
        P6 = strassenMatMult(np.subtract(B,D),np.add(G,H))
        P7 = strassenMatMult(np.subtract(A,C),np.add(E,F))

        # P5 + P4 - P2 + P6
        A = np.add(np.subtract(np.add(P5,P4),P2),P6)
        # P1 + P2
        B = np.add(P1,P2)
        # P3 + P4
        C = np.add(P3,P4)
        # P1 + P5 - P3 - P7
        D = np.subtract(np.subtract(np.add(P1,P5),P3),P7)

        #rejoins the smaller matricies
        matrix = np.array(np.concatenate([np.concatenate([A,B],axis=1),np.concatenate([C,D],axis=1)]))
        return matrix
    else:
        print("Something went wrong, sorry.")



#takes input of m1 (matrix 1) and m2 (matrix 2) and returns the product via Strassen-Winograd matrix multiplication
def SWMatMult(m1,m2):
    if (len(m1) == len(m2) == 2):
        # print("Base case!")
        matrix = np.array([
            # [ AE + BG ] , [ AF + BH]
            [m1[0][0] * m2[0][0] + m1[0][1] * m2[1][0], m1[0][0] * m2[0][1] + m1[0][1] * m2[1][1]],
            # [ CE + DG ] , [ CF + DH]
            [m1[1][0] * m2[0][0] + m1[1][1] * m2[1][0], m1[1][0] * m2[0][1] + m1[1][1] * m2[1][1]]
        ])
        # print(matrix)
        return matrix
    elif (len(m1) == len(m2)):
        splitSize = int(len(m1)/2)
        # print(splitSize)
        #slices the arrays more easily than classicMatMult
        A, B, C, D = m1[:splitSize, :splitSize], m1[:splitSize, splitSize:], m1[splitSize:, :splitSize], m1[splitSize:, splitSize:]
        E, F, G, H = m2[:splitSize, :splitSize], m2[:splitSize, splitSize:], m2[splitSize:, :splitSize], m2[splitSize:, splitSize:]
        # print(A,B,C,D)
        # print(E,F,G,H)

        S1 = np.add(C,D)
        S2 = np.subtract(S1,A)
        S3 = np.subtract(A,C)
        S4 = 

        P1 = SWMatMult(A,np.subtract(F,H))
        P2 = SWMatMult(np.add(A,B),H)
        P3 = SWMatMult(np.add(C,D),E)
        P4 = SWMatMult(D,np.subtract(G,E))
        P5 = SWMatMult(np.add(A,D),np.add(E,H))
        P6 = SWMatMult(np.subtract(B,D),np.add(G,H))
        P7 = SWMatMult(np.subtract(A,C),np.add(E,F))

        # P1 + P2
        A = np.add(np.subtract(np.add(P5,P4),P2),P6)
        # U3 + P3
        B = np.add(P1,P2)
        # U2 - P4
        C = np.add(P3,P4)
        # U2 + P6
        D = np.subtract(np.subtract(np.add(P1,P5),P3),P7)

        #rejoins the smaller matricies
        matrix = np.array(np.concatenate([np.concatenate([A,B],axis=1),np.concatenate([C,D],axis=1)]))
        return matrix
    else:
        print("Something went wrong, sorry.")

if __name__ == "__main__":
    # matrix = importMatrix("smallm.m")
    # x = np.indices((128,128),dtype=np.int)
    x = np.indices((8,8),dtype=np.int)
    print(x[0])
    # matrix = importMatrix("matrix.m")
    # print(matrix)
    # identity = importMatrix("identity.m")
    # print(classicMatMult(x[0],x[0]))
    # print(classicMatMult(matrix,identity))
    # print(classicMatMult(matrix,matrix))
    # print(strassenMatMult(matrix,matrix))
    print(strassenMatMult(x[0],x[0]))
    print(SWMatMult(x[0],x[0]))
    # print(SWMatMult(matrix,matrix))
