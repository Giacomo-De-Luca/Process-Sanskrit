
#from dotenv import load_dotenv

## If heroku postgres
#load_dotenv()
#DATABASE_URL = os.environ['DATABASE_URL']

#if local postgres

#DATABASE_URL = "postgresql+psycopg2://postgres:again@localhost:5432/sanskritmagicdb"
#DATABASE_URL = os.getenv("DATABASE_URL")
#DATABASE_URL = os.getenv("postgres://u5o7c326q19pvp:pdb08c4f74fd1c63df61a559fc3d7a261ac3c65a24df6cfd06fbb2ad511143f0d@c3cj4hehegopde.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/d61ijmaljbh829")

#if DATABASE_URL.startswith("postgres://"):
    #DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

#if using SQLite