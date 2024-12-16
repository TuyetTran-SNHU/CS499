count = 0
user_name= input("enter your username?")
while (count<3):
    user_pick=input("which do you choose: rock, paper, scissor?")
    import random
    computer_pick=random.randint(1,3)
    if user_pick== "rock":
        if computer_pick== 1:
            count=count
            print("it a tie")
        elif computer_pick==2:
            count= 0
            print ("lose")
        else:
            count = count+1
            print("win")
    elif user_pick== "paper":
        if computer_pick== 1:
            count=count+1
            print("win")
        elif computer_pick==2:
            count= count
            print ("tie")
        else:
            count =0
            print("lose")
    elif user_pick== "scissor":
        if computer_pick== 1:
            count=0
            print("lose")
        elif computer_pick==2:
            count= count+1
            print ("win")
        else:
            count = count
            print("tie")
    else:
        print("wrong input")
print ("you won 3x, game will end here")
        
        
