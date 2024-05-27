from sqlalchemy import MetaData, Table, Column
from data_utils import get_mlb_db_engine

def add_primary_keys(table_name, primary_keys):

    engine = get_mlb_db_engine()
    metadata = MetaData()

    #get old table
    old_table = Table(table_name, metadata, autoload_with=engine)

    #gen columns
    new_columns = []

    # get cols
    for col in old_table.columns:
        if col.name in primary_keys:
            new_col = Column(col.name, col.type, primary_key=True)
        else:
            new_col = Column(col.name, col.type, primary_key=False)
        new_columns.append(new_col)

    new_table = Table(
            f'{table_name}_new',
            metadata,
            *new_columns
        )

    if engine:
        new_table.create(engine)

if __name__ == '__main__':
    statcast_primary_keys = [
            'game_date',
            'game_pk',
            'at_bat_number',
            'pitch_number'
        ]
    add_primary_keys('Statcast', statcast_primary_keys)
    

