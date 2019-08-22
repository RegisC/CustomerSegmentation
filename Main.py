# Standard libraries
import argparse
import sys

# External libraries
import pandas as pd
import pickle
import sklearn.cluster

# Constants
MODEL_FILE = "P5-Model.sav"


def calc_client_stats(df, verbose=False):
    """Return client statistics as a DataFrame."""
    # Compute monetary value for each row
    df['Value'] = df['Quantity'] * df['UnitPrice']
    # Aggregation
    df1 = df[['CustomerID', 'InvoiceNo', 'Quantity', 'Value', 'UnitPrice']]
    clients = df1.groupby(['CustomerID']).agg(MeanQty=('Quantity', 'mean'),
                                              MaxQty=('Quantity', 'max'),
                                              MeanPrice=('UnitPrice', 'mean'),
                                              MinPrice=('UnitPrice', 'min'),
                                              MaxPrice=('UnitPrice', 'max'))
    # 1st level of aggregation
    cols = ['CustomerID', 'InvoiceNo']
    per_order = df1.groupby(cols).agg(OrderLen=('Quantity', 'count'))
    per_order.reset_index(inplace=True)
    # 2nd level
    cols = ['CustomerID']
    per_order = per_order.groupby(cols).agg(MeanOrderLen=('OrderLen', 'mean'))
    per_order.reset_index(inplace=True)
    # Join
    data = pd.merge(left=clients, right=per_order, on='CustomerID', how='left')
    if verbose:
        print("Statistiques client :")
        print(data)
    return data


def process_order(input_file, verbose=False):
    """Process order contained in `input_file`."""

    # Load invoice description file
    df = pd.read_excel(input_file)
    if verbose:
        print(f"Fichier {input_file} chargé contenant {len(df)} entrées.")
    # Calculate client stats and remove CustomerID column
    client_stats = calc_client_stats(df, verbose)
    cols = list(client_stats.columns)
    if verbose:
        print(f"Colonnes : {cols}")
    cols.remove('CustomerID')
    X = client_stats[cols]
    # Load model and determine the partition that the client belongs to
    load = pickle.load(open(MODEL_FILE, "rb"))
    model = load['model']
    trans = load['transformer']
    categories = load['categories']
    cat = model.predict(trans.transform(X))[0]
    print(f"Catégorie du client : {categories[cat]}")


def main():
    """Main routine."""

    p = argparse.ArgumentParser(description="Détermine la catégorie d'un "
                                            "client")
    p.add_argument('input_file', action='store',
                   help='Fichier contenant la commande du client')
    p.add_argument('-v', '--verbose', action='store_true', default=False,
                   dest='verbose', help='Active le mode verbeux')
    args = p.parse_args()
    if args.verbose:
        print(f"Fichier d'entrée : {args.input_file}")
    process_order(args.input_file, args.verbose)


# Script entry point
if __name__ == "__main__":
    main()
