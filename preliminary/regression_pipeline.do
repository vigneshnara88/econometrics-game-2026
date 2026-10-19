* this stata code was run on `final_clean_dataset.csv'. this is the final dataset that is cleaned to remove null values and noisy data (e.g. gpa_year1==0). a version is attached in our zip file.

describe
summarize gpa0 gpa_year1 gpa_year2 sex mtongue numcourses_nov1 lastmin mom_edn dad_edn d z g2 g3 z_g2 z_g3
tabstat gpa0 gpa_year1 gpa_year2 d z, by(hsgroup) statistics(count mean sd min max)

* check randomization balance (offer rates by group)
tab hsgroup z, freq row


* control variables: female, english, numcourses_nov1, lastmin, mom_edn, dad_edn
local controls "female english numcourses_nov1 lm_never lm_rarely mom1 mom2 dad1 dad2 gpa0"

* (1) first stage (correlation between offer and sfp)
reg d z g2 g3 z_g2 z_g3 `controls', robust


if _rc == 0 {
    estimates store fs_full
    
    * Calculate group-specific first-stage effects
	display ""
    display "First Stage Effects (by group):"
    display "Group 1 (baseline): " %6.4f _b[z] " (SE: " %6.4f _se[z] ")"
    
    lincom z + z_g2
    display "Group 2 effect: " %6.4f r(estimate) " (SE: " %6.4f r(se) ")"
    
    lincom z + z_g3
    display "Group 3 effect: " %6.4f r(estimate) " (SE: " %6.4f r(se) ")"
    
    * Test if first-stage varies by group
    display ""
    display "Tests for heterogeneity in first stage:"
    test z_g2 = 0
    display "H0: Group 2 = Group 1, p-value = " %6.4f r(p)
    
    test z_g3 = 0
    display "H0: Group 3 = Group 1, p-value = " %6.4f r(p)
    
    test z_g2 = z_g3
    display "H0: Group 2 = Group 3, p-value = " %6.4f r(p)
}

* (2) reduced form (effect of offer on outcome)
* gpa year 1
reg gpa_year1 z g2 g3 z_g2 z_g3 `controls', robust

if _rc == 0 {
    estimates store rf_gpa1
    
    display ""
    display "Group-Specific ITT Effects on GPA Year 1:"
    display "Group 1: " %6.4f _b[z] " (SE: " %6.4f _se[z] ")"
    
    lincom z + z_g2
    display "Group 2: " %6.4f r(estimate) " (SE: " %6.4f r(se) ")"
    
    lincom z + z_g3
    display "Group 3: " %6.4f r(estimate) " (SE: " %6.4f r(se) ")"
}

* gpa year 2
reg gpa_year2 z g2 g3 z_g2 z_g3 `controls', robust
if _rc == 0 {
    estimates store rf_gpa2
    
    display ""
    display "Group-Specific ITT Effects on GPA Year 2:"
    display "Group 1: " %6.4f _b[z] " (SE: " %6.4f _se[z] ")"
    
    lincom z + z_g2
    display "Group 2: " %6.4f r(estimate) " (SE: " %6.4f r(se) ")"
    
    lincom z + z_g3
    display "Group 3: " %6.4f r(estimate) " (SE: " %6.4f r(se) ")"
}


* iv (2sls)
* gpa year 1 
display "--- IV Regression: GPA Year 1 ---"
ivregress 2sls gpa_year1 `controls' (d d_g2 d_g3 = z z_g2 z_g3), robust first
if _rc == 0 {
    estimates store iv_gpa1
   
    display "Complier average causal effect on GPA Year 1:"

    display "Group 1 (Lowest HS GPA):  " %6.4f _b[d] " (SE: " %6.4f _se[d] ")"
    scalar late1 = _b[d]
    
    lincom d + d_g2
    display "Group 2 (Middle HS GPA):  " %6.4f r(estimate) " (SE: " %6.4f r(se) ")"
    scalar late2 = r(estimate)
    
    lincom d + d_g3
    display "Group 3 (Highest HS GPA): " %6.4f r(estimate) " (SE: " %6.4f r(se) ")"
    scalar late3 = r(estimate)
    
    * test heterogeneity 
    display ""
    display "Tests for heterogeneous treatment effects:"
    test d_g2 = 0
    display "Group 2 vs Group 1: p-value = " %6.4f r(p)
    
    test d_g3 = 0
    display "Group 3 vs Group 1: p-value = " %6.4f r(p)
    
    test d_g2 = d_g3
    display "Group 2 vs Group 3: p-value = " %6.4f r(p)
}

* gpa year 2
display "--- IV Regression: GPA Year 2 ---"
ivregress 2sls gpa_year2 `controls' (d d_g2 d_g3 = z z_g2 z_g3), robust
if _rc == 0 {
    estimates store iv_gpa2
    
    display ""
    display "Complier average causal effect on GPA Year 2:"

    display "Group 1 (Lowest HS GPA):  " %6.4f _b[d] " (SE: " %6.4f _se[d] ")"
    
    lincom d + d_g2
    display "Group 2 (Middle HS GPA):  " %6.4f r(estimate) " (SE: " %6.4f r(se) ")"
    
    lincom d + d_g3
    display "Group 3 (Highest HS GPA): " %6.4f r(estimate) " (SE: " %6.4f r(se) ")"
	
	* test heterogeneity 
    display ""
    display "Tests for heterogeneous treatment effects:"
    test d_g2 = 0
    display "Group 2 vs Group 1: p-value = " %6.4f r(p)
    
    test d_g3 = 0
    display "Group 3 vs Group 1: p-value = " %6.4f r(p)
    
    test d_g2 = d_g3
    display "Group 2 vs Group 3: p-value = " %6.4f r(p)
}
