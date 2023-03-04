import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Create a connection to the database
engine = create_engine('sqlite:///database.db')

# Create a session factory
Session = sessionmaker(bind=engine)

# Create a base class for our models
Base = declarative_base()

# app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(app)
# Base = declarative_base()


# Define a database model
class Inspection(Base):
    __tablename__ = 'inspection'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    threshold = Column(Integer)
    inference = Column(Integer)
    date_created = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<Inspection %r>' % self.id


# class Setting(Model):
#     id = Column(Integer, primary_key=True)
#     threshold = Column(Real, nullable=False)
#     inference = Column(Real, default=datetime.utcnow)
#     date_created = Column(DateTime, default=datetime.utcnow)

#     def __repr__(self):
#         return '<Setting %r>' % self.id


# def create_database(app):
#     if not os.path.exists('database.db'):
#         Base.metadata.create_all(engine)
#         # create_all(app=app)
#         print('database created!')
#     else:
#         print('database exists')

if __name__ == "__main__":
    if os.path.exists('database.db'):
        print('database exists')
    else:
        # Create the database tables
        Base.metadata.create_all(engine)
        print('database created!')
