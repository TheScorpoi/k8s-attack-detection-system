import pandas as pd

class PreProcessData:
    """
    Pre-process data for feature store ingestion.
    
    This class is responsible for pre-processing data before it is ingested into the feature store.
    The class have methods that convert data to a format that is suitable for the models and for the feature store itself.
    It is used by the upload_data.py script. 
    
    """

    def __init__(self, data: pd.DataFrame):
        """
        Constructor for the PreProcessData class.

        Args:
            data (pd.DataFrame): The input data to be pre-processed.
        """
        self.data = data
        
    def arrange_columns_names(self):
        """
        Arrange columns names.
        """
        self.data.columns = self.data.columns.str.lstrip('_')
        self.data.rename(columns={'source_@timestamp': 'source_timestamp'}, inplace=True)
        self.data = self.data.dropna(subset=['source_network_transport'])

        return self
        
    def network_transport_to_numerical(self):
        """
        Convert network transport to numerical values.

        """
        self.data = self.data[(self.data['source_network_transport'] != 'icmp') & (self.data['source_network_transport'] != 'ipv6-icmp')]
        self.data['source_network_transport'].replace({'tcp': 1, 'udp': 2}, inplace=True)
        
        return self

    def ip_address_to_numerical(self):
        """
        Convert IP addresses to numerical values.

        """

        def _ip_to_numerical(ip):
            """
            Convert IP addresses (IPv4 and IPv6) to numerical values.
            For IPv6, use a hash to ensure the value doesn't exceed 64 bits.

            Args:
                ip (str): ip address

            Returns:
                int: ip address converted to int or NaN
            """
            import ipaddress
            import numpy as np
            import hashlib

            try:
                ip_obj = ipaddress.ip_address(ip)
                if type(ip_obj) is ipaddress.IPv4Address:
                    return int(ip_obj)
                elif type(ip_obj) is ipaddress.IPv6Address:
                    # Use a hash to convert IPv6 to a shorter fixed length
                    hash_digest = hashlib.sha256(ip.encode()).digest()
                    # Convert the first 8 bytes to an integer
                    return int.from_bytes(hash_digest[:8], 'big')
            except ValueError:
                pass
            return np.nan

        self.data['source_source_ip'] = self.data['source_source_ip'].apply(_ip_to_numerical)
        self.data['source_destination_ip'] = self.data['source_destination_ip'].apply(_ip_to_numerical)

        self.data = self.data.dropna(subset=['source_source_ip', 'source_destination_ip'])

        return self

    def source_flow_final_to_numerical(self):
        """
        Convert source flow final to numerical values.

        """
        self.data['source_flow_final'] = self.data['source_flow_final'].map({False: 0, True: 1})
        
        return self

    def source_flow_id_to_numerical(self):
        """
        Convert source flow ID to numerical values.

        """
        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()
        self.data['source_flow_id_encoded'] = label_encoder.fit_transform(self.data['source_flow_id'])
        self.data = self.data.drop('source_flow_id', axis=1)
        self.data = pd.concat([self.data['source_flow_id_encoded'], self.data.drop('source_flow_id_encoded', axis=1)], axis=1)
        
        return self

    def convert_timestamp_to_numerical(self):
        """
        Convert timestamp to numerical values.

        """
        
        self.data['source_timestamp'] = pd.to_datetime(self.data['source_timestamp'])
        self.data['source_timestamp'] = self.data['source_timestamp'].astype(int) // 10**6
        
        return self
    

    def convert_uint64_to_int(self):
        """
        Convert uint64 to int.

        """
        
        self.data['source_source_ip'] = self.data['source_source_ip'].astype(int)
        self.data['source_destination_ip'] = self.data['source_destination_ip'].astype(int)
        self.data['source_timestamp'] = pd.to_datetime(self.data['source_timestamp']).astype(int) / 10**9

        return self
        
        

##########################
if __name__ == "__main__":
    #just for testing purposes
    data = pd.read_csv("../data/629887f7-f6aa-4d77-b0db-83822a92c582_1_all/elastic_february2022_data.csv")
    
    preprocess_data = PreProcessData(data)
    result = preprocess_data.arrange_columns_names().network_transport_to_numerical().ip_address_to_numerical().source_flow_final_to_numerical().source_flow_id_to_numerical().convert_timestamp_to_numerical()