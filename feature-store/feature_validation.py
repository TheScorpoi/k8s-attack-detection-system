import pandas as pd
import great_expectations as ge
from great_expectations.core import ExpectationSuite, ExpectationConfiguration

class FeatureValidation():
    """
    Feature validation class.
    This class is responsible for validating data before it is ingested into the feature store.
    
    """

    def __init__(self, data: pd.DataFrame):
        """

        Args:
            data (pd.DataFrame): _description_
        """
        self.data = data
        
    def _validate_data(self):
        """
        Validate data through Great Expectations library, just a simple set of validations like min and max values.

        Returns:
            expectation_suite_trans (_type_): Great Expectations expectation suite.
        """
        ge_trans_df = ge.from_pandas(self.data)
        
        expectation_suite_trans = ge_trans_df.get_expectation_suite()
        expectation_suite_trans.expectation_suite_name = "validade_traffic_data"
        print(expectation_suite_trans)
        
        
        expectation_suite_trans.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_min_to_be_between",
                kwargs={
                    "column":"source_flow_final",
                    "min_value": 0
                }
            )  
        )

        expectation_suite_trans.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_min_to_be_between",
                kwargs={
                    "column":"source_flow_final",
                    "min_value": 17,
                    "max_value": 130
                }
            )
        )
        
        expectation_suite_trans.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column":"source_network_bytes"}
            )
        )
        
        ge_trans_df = ge.from_pandas(self.data, expectation_suite=expectation_suite_trans)
        validation_report_trans = ge_trans_df.validate()
        print(validation_report_trans)
        expectation_suite_profiled, validation_report = ge_trans_df.profile(profiler=ge.profile.BasicSuiteBuilderProfiler)
        print(f"The suite contains {len(expectation_suite_profiled['expectations'])} expectations for {len(self.data.columns.values)} columns. See sample below\n" + ge_trans_df.get_expectation_suite().__repr__()[:455])

        return expectation_suite_trans

    def register_expectation_suite(self, feature_group):
        """
        Register expectation suite in the feature store, and call validate_data method.

        Args:
            feature_gruup (_type_): _description_
            
        """

        expectation_suite_trans = self._validate_data()
        feature_group.save_expectation_suite(expectation_suite_trans, validation_ingestion_policy="ALWAYS")
        