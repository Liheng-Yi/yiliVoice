rating = [2,5,3,4,1]
n = len(rating)

# Dynamic programming tables
inc = [0] * n
dec = [0] * n

# Count valid 2-element sequences ending at each index
for j in range(n):
    for i in range(j):
        if rating[i] < rating[j]:
            inc[j] += 1
            print(inc)
        elif rating[i] > rating[j]:
            dec[j] += 1

print(inc, dec)

# Count valid 3-element sequences ending at each index
count_inc = 0
count_dec = 0
for k in range(n):
    for j in range(k):
        if rating[j] < rating[k]:
            count_inc += inc[j]  # Sequences ending at j can form a 3-seq ending at k
        if rating[j] > rating[k]:
            count_dec += dec[j]  # Sequences ending at j can form a 3-seq ending at k

print(count_inc + count_dec)

