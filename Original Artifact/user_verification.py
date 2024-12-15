import access_pymongo

client = access_pymongo.client

# client is initialized in access_pymongo - 2 different method
'''
MongoDB = access_pymongo.client.userdb # Database reference
users_collection = MongoDB.users  # Collection reference'''
db = client["db"]  # Adjust "db" to your actual database name
collection = db["users"]

def user():
    choice = input("Press 1 for returning user or 2 for new user: ")

    if choice == "1":
        # Code for returning user
        print("Returning user")
        attempts = 0
        while attempts < 3:
            username = input('Enter Username: ')
            result = collection.find_one({"username": username})
            
            if result:
                print("Connected")
                return username  # Corrected return statement
            else:
                print("Not in the system")
                attempts += 1

                if attempts == 3:
                    print("3 Attempts")
                    break

    elif choice == "2":
        # Code for new user
        print("New user")
        new_user = input('Enter Username: ')
        
        # Validate the username (less than 26 characters, alphanumeric)
        if len(new_user) >= 26 or not new_user.isalnum():
            print("Username must be less than 26 characters long and contain only letters and numbers.")
            return
        
        # Check if username already exists in the database
        existing_user = collection.find_one({"username": new_user})
        if existing_user:
            print("Please choose a different username.")
            return
        
        # Insert new user into the database
        post = {'username': new_user}
        try:
            collection.insert_one(post)  # Corrected collection name
            print("New user added successfully.")
            return new_user  # Return the newly created username
        except Exception as e:
            print(f"Error occurred: {e}")

    else:
        print("Invalid input")