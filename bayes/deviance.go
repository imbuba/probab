// Bayesian deviance
// Aitkin 2010: 41

package bayes

import "math"

// Bayesian deviance
//nolint:unused
func deviance(likelihood float64) float64 {
	return -2 * math.Log(likelihood)
}

// Bayesian deviance difference
//nolint:unused
func devdiff(like1, like2 float64) float64 {
	lr := like1 / like2
	return -2 * math.Log(lr)
}
