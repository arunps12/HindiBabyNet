source(file.path("..", "R", "00_setup.R"))
source(file.path("..", "R", "functions", "model_helpers.R"))

testthat::test_that("family 1 formula matches the configured structure", {
  formula <- build_family_1_formula("median_F0_Hz")
  formula_text <- paste(deparse(formula), collapse = " ")
  testthat::expect_true(grepl("median_F0_Hz ~", formula_text, fixed = TRUE))
  testthat::expect_true(grepl("speaker", formula_text, fixed = TRUE))
  testthat::expect_true(grepl("age_z", formula_text, fixed = TRUE))
  testthat::expect_true(grepl("SES_num", formula_text, fixed = TRUE))
  testthat::expect_true(grepl("speaker:age_z", formula_text, fixed = TRUE))
  testthat::expect_true(grepl("(1 | participant_id)", formula_text, fixed = TRUE))
})

testthat::test_that("family 2 formula excludes input by speaker interaction", {
  formula <- build_family_2_formula("key_child_count_per_hour", "median_F0_Hz_z", "input_count_per_hour", "family_2_count")
  formula_text <- paste(deparse(formula), collapse = " ")
  testthat::expect_true(grepl("median_F0_Hz_z * speaker", formula_text, fixed = TRUE))
  testthat::expect_true(grepl("input_count_per_hour", formula_text, fixed = TRUE))
  testthat::expect_true(grepl("SES_num", formula_text, fixed = TRUE))
  testthat::expect_false(grepl("input_count_per_hour:speaker", formula_text, fixed = TRUE))
})

testthat::test_that("continuous predictors are standardized into z columns", {
  dataframe <- data.frame(median_F0_Hz = c(100, 200, 300))
  result <- standardize_predictor(dataframe, "median_F0_Hz")
  testthat::expect_equal(result$standardized_name, "median_F0_Hz_z")
  testthat::expect_equal(round(mean(result$data$median_F0_Hz_z), 10), 0)
})

testthat::test_that("model stub names remain unique across families and predictors", {
  stub1 <- build_model_stub("family1", "median_F0_Hz")
  stub2 <- build_model_stub("family_2_count", "key_child_count_per_hour", "median_F0_Hz")
  testthat::expect_false(identical(stub1, stub2))
})
