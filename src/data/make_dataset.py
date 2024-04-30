import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import sqlite3
import json

logger = logging.getLogger(__name__)
DB_LOCATION = '../../data/databases/mlb.db'

def create_table(table_name = 'Statcast'):
    """Creates tables in data/databases/mlb.db for table schemas found in the table_schema.json file

    Args:
        table_name (str, optional): name of table to initialize from table_schema.json file. Defaults to 'Statcast'.
    """
    
    table_schemas_path = '../../data/databases/table_schema.json'
    DB_LOCATION = '../../data/databases/mlb.db'

    if Path(table_schemas_path).exists():
        
        # read schema/create table script
        with open(table_schemas_path, 'r') as file:
            schema = json.load(file)

        # validate table schema exists
        if table_name in schema:
            table_schema = schema[table_name]
        else:
            logger.critical(f'no {table_name} schema found in table_schema.json')
            exit()


        # initialize db
        conn = sqlite3.connect(DB_LOCATION)
        c = conn.cursor()
        
        # generate create table string from schema
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ("
        for column in table_schema['columns']:
            create_table_sql += f"{column['name']} {column['type']}, "
        # add the primary key constraint
        if 'primaryKey' in table_schema:
            create_table_sql += f"PRIMARY KEY ({', '.join(table_schema['primaryKey'])})"
        create_table_sql += ")"

        # execute the SQL statement
        c.execute(create_table_sql)

        conn.commit()
        conn.close()

        logger.info(f'Success: "{table_name}"')
    
    else:
        logger.critical(f'table_schema.json not found at {table_schemas_path}')
        exit()
    

def main():
    logger.info('Starting SQLite db creation/updates')

    #make sure all desired tables are created
    required_tables = ['Statcast']
    for table in required_tables:
        logger.info(f'Validating "{table}" exists')
        create_table(table)
        logger.info(f'Updating "{table}"')

    #run updates on tables

    logger.info('DB creation and updates complete')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
