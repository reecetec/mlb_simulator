import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import sqlite3
import json
import pandas as pd
from pybaseball import statcast

logger = logging.getLogger(__name__)

def create_table(table_name = 'Statcast'):
    
    table_schemas_path = '../../data/databases/table_schema.json'
    db_location = '../../data/databases/mlb.db'

    if Path(table_schemas_path).exists():
        
        #read schema/create table script
        with open(table_schemas_path, 'r') as file:
            schema = json.load(file)

        if table_name in schema:
            table_schema = schema[table_name]
        else:
            logger.critical(f'no {table_name} schema found in table_schema.json')
            exit()


        #initialize db
        conn = sqlite3.connect(db_location)
        c = conn.cursor()
        
        #generate create table string from schema
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ("
        for column in table_schema['columns']:
            create_table_sql += f"{column['name']} {column['type']}, "
        #add the primary key constraint
        if 'primaryKey' in table_schema:
            create_table_sql += f"PRIMARY KEY ({', '.join(table_schema['primaryKey'])})"
        create_table_sql += ")"

        # Execute the SQL statement
        c.execute(create_table_sql)

        conn.commit()
        conn.close()

        logger.info(f'{table_name} successfully created')
    
    else:
        logger.critical(f'table_schema.json not found at {table_schemas_path}')
        exit()
    

def main():
    logger.info('starting SQLite db configuration for MLB_SIMULATOR')
    create_table()
    logger.info('finished')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())



    main()
