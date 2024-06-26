from cmath import sqrt


def Euclidian(point1, point2):
    if len(point1) != len(point2):
        return 0
    
    sum = 0
    for i in range(len(point1)):
        sum += (point1[i] - point2[i])*(point1[i] - point2[i])

    return sqrt(sum)


def Manhattan(point1, point2):
    if len(point1) != len(point2):
        return 0
    
    sum = 0
    for i in range(len(point1)):
        sum += (point1[i] - point2[i])

    return sum


def Cosine_Similarity(point1, point2):
    if len(point1) != len(point2):
        return 0

    dividend = 0
    divisor0 = 0
    divisor1 = 0
    divisor = 0
    for i in range(len(point1)):
        dividend += point1[i]*point2[i]
        
        divisor0 += point1[i]*point1[i]
        divisor1 += point2[i]*point2[i]

    divisor = sqrt(divisor0) * sqrt(divisor1)

    return dividend/divisor