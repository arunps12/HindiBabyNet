source(file.path("..", "R", "00_setup.R"))
source(file.path("..", "R", "functions", "table_helpers.R"))
source(file.path("..", "R", "functions", "data_helpers.R"))

testthat::test_that("identifier standardization trims and normalizes blanks", {
  values <- c(" P001 ", "", "NA", "P002")
  testthat::expect_equal(standardize_identifier(values), c("P001", NA, NA, "P002"))
})

testthat::test_that("speaker mapping converts codes to canonical labels", {
  speakers <- c("FEM", "MAL", "OCH", "adult_female")
  testthat::expect_equal(
    standardize_speaker_labels(speakers),
    c("adult_female", "adult_male", "other_child", "adult_female")
  )
})

testthat::test_that("duplicate key detection stops on repeated participant speaker rows", {
  dataframe <- data.frame(participant_id = c("P001", "P001"), speaker = c("adult_female", "adult_female"), stringsAsFactors = FALSE)
  testthat::expect_error(assert_unique_keys(dataframe, "dup", c("participant_id", "speaker")), "duplicate")
})

testthat::test_that("merge-key validation reports unmatched ids", {
  left <- data.frame(participant_id = c("P001", "P002"), stringsAsFactors = FALSE)
  right <- data.frame(participant_id = c("P001", "P003"), stringsAsFactors = FALSE)
  unmatched <- find_unmatched_ids(left, right, by_columns = "participant_id")
  testthat::expect_equal(unmatched$left_only$participant_id, "P002")
  testthat::expect_equal(unmatched$right_only$participant_id, "P003")
})

testthat::test_that("row-structure and repeated-outcome checks work on synthetic long data", {
  dataframe <- data.frame(
    participant_id = rep("P001", 3),
    speaker = c("adult_female", "adult_male", "other_child"),
    key_child_count_per_hour = c(1, 1, 1),
    key_child_duration_per_hour = c(2, 2, 2),
    input_count_per_hour = c(3, 4, 5),
    input_duration_per_hour = c(6, 7, 8),
    stringsAsFactors = FALSE
  )
  testthat::expect_error(assert_unique_keys(dataframe, "long", c("participant_id", "speaker")), NA)
  testthat::expect_error(check_repeated_outcomes_constant(dataframe), NA)
  variation <- build_variation_check_table(dataframe)
  testthat::expect_true(all(variation$status))
})

testthat::test_that("missing-data handling counts complete rows and participants", {
  dataframe <- data.frame(
    participant_id = c("P001", "P002"),
    speaker = c("adult_female", "adult_male"),
    age_z = c(0, NA),
    SES_num = c(2, 2),
    child_sex = c("female", "male"),
    city = c("CNB", "CNB"),
    median_F0_Hz = c(200, 250),
    stringsAsFactors = FALSE
  )
  counts <- count_complete_cases(dataframe, c("participant_id", "speaker", "age_z", "SES_num", "child_sex", "city", "median_F0_Hz"))
  testthat::expect_equal(counts$n_complete_rows, 1)
  testthat::expect_equal(counts$n_complete_participants, 1)
})

testthat::test_that("SES is converted to ordered numeric coding", {
  dataframe <- data.frame(SES = c("primary_school", "high_school", "master"), stringsAsFactors = FALSE)
  result <- add_continuous_ses(dataframe)
  testthat::expect_true(is.ordered(result$SES))
  testthat::expect_equal(result$SES_num, c(1, 2, 5))
})
