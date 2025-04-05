import sqlite3


## Connect to Sqllite
connection = sqlite3.connect("student.db")

## Create a Cursor object to insert record, create table
cursor = connection.cursor()

## Create table
table_info = """
CREATE TABLE STUDENT(
    NAME VARCHAR(25), CLASS VARCHAR(25),
    SECTION VARCHAR(25), MARKS INT
)
"""

cursor.execute(table_info)

## Insert Record
cursor.execute(''' INSERT into STUDENT values ('Saad', 'Data Science', 'A', 90)''')
cursor.execute(''' INSERT into STUDENT values ('Usman', 'Data Science', 'A', 95)''')
cursor.execute(''' INSERT into STUDENT values ('Arhum', 'Data Science', 'B', 80)''')
cursor.execute(''' INSERT into STUDENT values ('Kausain', 'Data Science', 'B', 92)''')
cursor.execute(''' INSERT into STUDENT values ('Haris', 'DevOps', 'A', 85)''')
cursor.execute(''' INSERT into STUDENT values ('Umer', 'Cyber Security', 'B', 90)''')

## Display all the records
print("The Inserted Records are")
data = cursor.execute('''SELECT * FROM STUDENT''')
for row in data:
    print(row)

## Commit changes and closing connection
connection.commit()
connection.close()
