# ============================================================
# Model reporting helper functions
# ============================================================

library(knitr)
library(kableExtra)
library(broom.mixed)
library(broom)
library(performance)
library(tibble)

#------------------------------------------------------------
# Find the fitted model inside an RDS object
#------------------------------------------------------------
get_actual_model <- function(x) {
  
  model_classes <- c(
    "lmerMod",
    "glmerMod",
    "lm",
    "glm",
    "glmmTMB"
  )
  
  if (inherits(x, model_classes))
    return(x)
  
  possible_names <- c(
    "model",
    "best_model",
    "final_model",
    "fitted_model",
    "fit"
  )
  
  if (is.list(x)) {
    
    ## First try common names
    for (nm in possible_names) {
      
      if (nm %in% names(x)) {
        
        obj <- x[[nm]]
        
        if (inherits(obj, model_classes))
          return(obj)
      }
    }
    
    ## Otherwise search every list element
    for (obj in x) {
      
      if (inherits(obj, model_classes))
        return(obj)
    }
  }
  
  stop("No fitted model found inside this RDS object.")
}

#------------------------------------------------------------
# Nice table
#------------------------------------------------------------
nice_table <- function(x, caption = NULL) {
  
  kable(
    x,
    digits = 3,
    caption = caption
  ) |>
    
    kable_styling(
      bootstrap_options = c(
        "striped",
        "hover",
        "condensed"
      ),
      full_width = FALSE,
      position = "left"
    )
}

#------------------------------------------------------------
# Report one model
#------------------------------------------------------------
report_one_model <- function(model_path, model_name) {
  
  obj <- readRDS(model_path)
  
  model <- get_actual_model(obj)
  
  cat("\n# ", model_name, "\n\n")
  
  #----------------------------------------------------------
  # Formula
  #----------------------------------------------------------
  
  cat("## Model formula\n\n")
  
  cat("```r\n")
  cat(paste(deparse(formula(model)), collapse = " "))
  cat("\n```\n\n")
  
  #----------------------------------------------------------
  # Model fit
  #----------------------------------------------------------
  
  fit_tbl <- tibble(
    
    AIC = AIC(model),
    
    BIC = BIC(model),
    
    LogLik = as.numeric(logLik(model))
  )
  
  print(
    
    nice_table(
      fit_tbl,
      "Model fit statistics"
    )
  )
  
  #----------------------------------------------------------
  # R2
  #----------------------------------------------------------
  
  r2_tbl <- tryCatch(as.data.frame(performance::r2(model)), error = function(e) data.frame(note = conditionMessage(e)))
  
  print(
    
    nice_table(
      r2_tbl,
      "Marginal and Conditional R²"
    )
  )
  
  #----------------------------------------------------------
  # ICC
  #----------------------------------------------------------
  
  icc_tbl <- tryCatch(as.data.frame(performance::icc(model)), error = function(e) data.frame(note = conditionMessage(e)))
  
  print(
    
    nice_table(
      icc_tbl,
      "Intraclass Correlation Coefficient"
    )
  )
  
  #----------------------------------------------------------
  # Fixed effects
  #----------------------------------------------------------
  
  fixed_tbl <- if (inherits(model, c("lmerMod", "glmerMod", "glmmTMB", "merMod"))) {
    broom.mixed::tidy(
      model,
      effects = "fixed",
      conf.int = TRUE
    )
  } else {
    broom::tidy(
      model,
      conf.int = TRUE
    )
  }
  
  print(
    
    nice_table(
      
      fixed_tbl,
      
      "Fixed effects"
    )
  )
}