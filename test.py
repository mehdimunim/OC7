import boto3

# Créez un client CloudWatch sans spécifier d'identifiants
client = boto3.client('cloudwatch')

# Essayez de lister les métriques CloudWatch
try:
    response = client.list_metrics()
    print("Connexion réussie!")
    # print(response)  # Décommentez pour afficher la réponse complète
except Exception as e:
    print(f"Erreur de connexion: {e}")
