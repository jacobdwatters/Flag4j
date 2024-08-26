/*
 * MIT License
 *
 * Copyright (c) 2024. Jacob Watters
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package org.flag4j.core_temp.structures.fields;

import org.flag4j.complex_numbers.ComplexNumberParser;
import org.flag4j.util.ErrorMessages;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * <p>A complex number stored in rectangular form and backed by two 64-bit floating point number (one for the real and one for the
 * imaginary component).</p>
 *
 * <p>Complex128 numbers are Immutable.</p>
 */
public class Complex128 implements Field<Complex128> {
    // Several constants are provided for convenience.
    /**
     * The complex number with zero imaginary and real parts.
     */
    public static final Complex128 ZERO = new Complex128(0);
    /**
     * The complex number with zero imaginary part and one real part.
     */
    public static final Complex128 ONE = new Complex128(1);
    /**
     * The complex number with zero imaginary part and two real part.
     */
    public static final Complex128 TWO = new Complex128(2);
    /**
     * The double value closer than any other to the square root of 3
     */
    public static final Complex128 ROOT_THREE = new Complex128(Math.sqrt(3));
    /**
     * The imaginary unit i.
     */
    public static final Complex128 IMAGINARY_UNIT = new Complex128(0, 1);
    /**
     * The additive inverse of the imaginary unit, -i.
     */
    public static final Complex128 INV_IMAGINARY_UNIT = new Complex128(0, -1);
    /**
     * The maximum real double value 1.7976931348623157E308.
     */
    public static final Complex128 MAX_REAL = new Complex128(Double.MAX_VALUE);
    /**
     * The minimum real double value 4.9E-324
     */
    public static final Complex128 MIN_REAL = new Complex128(Double.MIN_VALUE);
    /**
     * The smallest possible real normal double 2.2250738585072014E-308.
     */
    public static final Complex128 MIN_REAL_NORMAL = new Complex128(Double.MIN_NORMAL);
    /**
     * Complex number with real part equal to {@link Double#POSITIVE_INFINITY}.
     */
    public static final Complex128 POSITIVE_INFINITY = new Complex128(Double.POSITIVE_INFINITY);
    /**
     * Complex number with real part equal to {@link Double#NEGATIVE_INFINITY}.
     */
    public static final Complex128 NEGATIVE_INFINITY = new Complex128(Double.NEGATIVE_INFINITY);
    /**
     * Complex number with real and imaginary parts equal to {@link Double#NaN}.
     */
    public static final Complex128 NaN = new Complex128(Double.NaN, Double.NaN);
    
    /**
     * Real component of the complex number.
     */
    public final double re;
    /**
     * Imaginary component of the complex number.
     */
    public final double im;


    /**
     * Constructs a complex number with specified real component and zero imaginary component.
     * @param re Real component of complex number.
     */
    public Complex128(double re) {
        this.re = re;
        this.im = 0;
    }


    /**
     * Constructs a complex number with specified complex and real components.
     * @param re Real component of complex number.
     * @param im Imaginary component of complex number.
     */
    public Complex128(double re, double im) {
        this.re = re;
        this.im = im;
    }


    /**
     * Constructs a complex number from a string of the form {@code "a +/- bi"} where {@code a} and {b} are real values and either may be
     * omitted. i.e. {@code "a", "bi", "a +/- i"}, and {@code "i"} are all also valid.
     * @param num The string representation of a complex number.
     */
    public Complex128(String num) {
        Complex128 complexNum = ComplexNumberParser.parseNumberToComplex128(num);
        this.re = complexNum.re;
        this.im = complexNum.im;
    }


    /**
     * Constructs a 128-bit complex number from a 64-bit complex number.
     * @param num The 64-bit complex number.
     */
    public Complex128(Complex64 num) {
        this.re = num.re;
        this.im = num.im;
    }


    /**
     * Checks if an object is equal to this Field element.
     * @param b Object to compare to this Field element.
     * @return True if the objects are the same or are both {@link Complex128}'s and have equal real and imaginary parts.
     */
    @Override
    public boolean equals(Object b) {
        // Check for quick returns.
        if(this == b) return true;
        if(b == null) return false;
        if(b.getClass() != this.getClass()) return false;
        Complex128 bCmp = (Complex128) b;

        return this.re == bCmp.re && this.im == bCmp.im;
    }


    /**
     * Generates the hashcode for this complex number.
     * @return An integer hash for this complex number.
     */
    @Override
    public int hashCode() {
        final int hashPrime1 = 7;
        final int hashPrime2 = 31;

        int hash = hashPrime2*hashPrime1 + Double.hashCode(re);
        hash = hashPrime2*hash + Double.hashCode(im);
        return hash;
    }


    /**
     * Checks if this complex number has zero imaginary part.
     * @return True if this complex number has zero imaginary part. False otherwise.
     * @see #isComplex()
     * @see #isImaginary()
     */
    public boolean isReal() {
        return this.im == 0;
    }


    /**
     * Checks if this complex number has zero real part.
     * @return True if this complex number has zero real part. False otherwise.
     * @see #isComplex()
     * @see #isReal()
     */
    public boolean isImaginary() {
        return this.re == 0;
    }


    /**
     * Checks if this complex number has non-zero imaginary part.
     * @return True if this complex number has non-zero imaginary part. False otherwise.
     * @see #isReal()
     * @see #isImaginary()
     */
    public boolean isComplex() {
        return this.im != 0;
    }


    /**
     * Sums two elements of this field (associative and commutative).
     * @param b Second field element in sum.
     * @return The sum of this element and {@code b}.
     */
    @Override
    public Complex128 add(Complex128 b) {
        return new Complex128(re + b.re, im + b.im);
    }


    /**
     * Sums an elements of this field with a real number.
     * @param b Second element in sum.
     * @return The sum of this element and {@code b}.
     */
    public Complex128 add(double b) {
        return new Complex128(re + b, im);
    }


    /**
     * Computes difference of two elements of this field.
     * @param b Second field element in difference.
     * @return The difference of this field element and {@code b}.
     */
    @Override
    public Complex128 sub(Complex128 b) {
        return new Complex128(re - b.re, im - b.im);
    }


    /**
     * Computes difference of an element of this field and a real number.
     * @param b Second element in difference.
     * @return The difference of this field element and {@code b}.
     */
    public Complex128 sub(double b) {
        return new Complex128(re - b, im);
    }


    /**
     * Computes the sum of all entires of specified array.
     * @param values Values to compute product of.
     * @return The sum of all values in {@code values}.
     */
    public static Complex128 sum(Complex128... numbers) {
        double re = 0;
        double im = 0;

        for(Complex128 num : numbers) {
            re += num.re;
            im += num.im;
        }

        return new Complex128(re, im);
    }


    /**
     * Multiplies two elements of this field (associative and commutative).
     * @param b Second field element in product.
     * @return The product of this field element and {@code b}.
     */
    @Override
    public Complex128 mult(Complex128 b) {
        return new Complex128(
                this.re*b.re - this.im*b.im,
                this.re*b.im + this.im*b.re
        );
    }


    /**
     * <p>Checks if this value is an additive identity for this semi-ring.</p>
     *
     * <p>An element 0 is an additive identity if a + 0 = a for any a in the semi-ring.</p>
     *
     * @return True if this value is an additive identity for this semi-ring. Otherwise, false.
     */
    @Override
    public boolean isZero() {
        return equals(ZERO);
    }


    /**
     * <p>Checks if this value is a multiplicitive identity for this semi-ring.</p>
     *
     * <p>An element 1 is a multiplicitive identity if a * 1 = a for any a in the semi-ring.</p>
     *
     * @return True if this value is a multiplicitive identity for this semi-ring. Otherwise, false.
     */
    @Override
    public boolean isOne() {
        return equals(ONE);
    }


    /**
     * <p>Gets the additive identity for this semi-ring.</p>
     *
     * <p>An element 0 is an additive identity if a + 0 = a for any a in the semi-ring.</p>
     *
     * @return The additive identity for this semi-ring.
     */
    @Override
    public Complex128 getZero() {
        return ZERO;
    }


    /**
     * <p>Gets the multiplicitive identity for this semi-ring.</p>
     *
     * <p>An element 1 is a multiplicitive identity if a * 1 = a for any a in the semi-ring.</p>
     *
     * @return The multiplicitive identity for this semi-ring.
     */
    @Override
    public Complex128 getOne() {
        return ONE;
    }


    /**
     * Multiplies an element of this field with a real number.
     * @param b Second element in product.
     * @return The product of this field element and {@code b}.
     */
    public Complex128 mult(double b) {
        return new Complex128(re*b, im*b);
    }


    /**
     * Computes the quotient of two elements of this field.
     * @param b Second field element in quotient.
     * @return The quotient of this field element and {@code b}.
     */
    @Override
    public Complex128 div(Complex128 b) {
        Complex128 quotient;

        if (this.equals(ZERO) && !b.equals(ZERO)) {
            quotient = ZERO;

        } else if(b.isReal() && !b.equals(ZERO)) {
            quotient = new Complex128(re/b.re, im/b.re);

        } else {
            double divisor = b.re*b.re + b.im*b.im;

            quotient = new Complex128(
                    (re*b.re + im*b.im) / divisor,
                    (im*b.re - re*b.im) / divisor);
        }

        return quotient;
    }


    /**
     * Computes the quotient of an elements of this field and a real number.
     * @param b Second element in quotient.
     * @return The quotient of this field element and {@code b}.
     */
    public Complex128 div(double b) {
        if(this.equals(ZERO) && b != 0) return ZERO;
        else return new Complex128(re/b, im/b);
    }


    /**
     * Computes the magnitude of this field element.
     * @return The magniitude of this field element.
     */
    @Override
    public double mag() {
        double absRe = Math.abs(this.re);
        double absIm = Math.abs(this.im);
        double max;
        double min;

        if(absRe > absIm) {
            max = absRe;
            min = absIm;
        } else {
            max = absIm;
            min = absRe;
        }

        if(max == 0.0) {
            return 0.0;
        }

        double ratio = min / max;
        return max * Math.sqrt(1 + ratio*ratio);
    }


    /**
     * Computes the square root of this field element.
     *
     * @return The square root of this field element.
     */
    @Override
    public Complex128 sqrt() {
        Complex128 result;

        if(isReal() && re>=0) {
            result = new Complex128(Math.sqrt(re));

        } else if(isReal() && re<0) {
            result = new Complex128(0, Math.sqrt(-re));

        } else {
            double mag = mag();
            double factor = im / Math.abs(im);

            double sqrtrRe = Math.sqrt((mag + this.re)/2);
            double sqrtIm = factor*Math.sqrt((mag - this.re)/2);

            result = new Complex128(sqrtrRe, sqrtIm);
        }

        return result;
    }


    /**
     * <p>Squares this magnitude of this complex number.</p>
     *
     * @return The square of the magnitude of this complex number as a real value.
     */
    public double magSquared() {
        return re*re + im*im;
    }


    /**
     * <p>Computes the addative inverse for an element of this field.</p>
     *
     * <p>An element -x is an addative inverse for a filed element x if -x + x = 0 where 0 is the addative identity..</p>
     *
     * @return The additive inverse for this field element.
     */
    public Complex128 addInv() {
        return new Complex128(-re, -im);
    }


    /**
     * Computes the multiplicative inverse of this complex number.
     * @return The multiplicative inverse of this complex number.
     */
    public Complex128 multInv() {
        return ONE.div(this);
    }


    /**
     * Computes the conjugate of this field element.
     * @return The conjugate of this field element.
     */
    @Override
    public Complex128 conj() {
        return new Complex128(re, -im);
    }


    /**
     * Compute a raised to the power of b. This method wraps {@link Math#pow(double, double)}
     * and returns a {@link Complex128}.
     * @param a The base.
     * @param b The exponent.
     * @return a raised to the power of b.
     */
    public static Complex128 pow(double a, double b) {
        return new Complex128(Math.pow(a, b));
    }


    /**
     * Compute a raised to the power of b.
     * and returns a {@link Complex128}.
     * @param a The base.
     * @param b The exponent.
     * @return a to the power of b.
     */
    public static Complex128 pow(double a, Complex128 b) {
        Complex128 result;

        if(b.im == 0) {
            result = Complex128.pow(a, b.re);
        } else {
            // Apply a change of base
            double logA = Math.log(a);
            Complex128 power = new Complex128(logA * b.re, logA * b.im);
            result = exp(power);
        }

        return result;
    }


    /**
     * Compute a raised to the power of b.
     * and returns a {@link Complex128}.
     * @param a The base.
     * @param b The exponent.
     * @return a to the power of b.
     */
    public static Complex128 pow(Complex128 a, Complex128 b) {
        Complex128 result;

        if(a.im == 0) {
            if(b.im == 0) {
                result = Complex128.pow(a.re, b.re);
            } else {
                result = Complex128.pow(a.re, b);
            }
        } else {
            result = exp(b.mult(ln(a)));
        }

        return result;
    }


    /**
     * Computes {@code a} raised to the power of {@code b}.
     * @param a The base.
     * @param b The exponent.
     * @return {@code a} raised to the power of {@code b}.
     */
    public static Complex128 pow(Complex128 a, double b) {
        Complex128 result;

        if(a.im == 0) {
            result = Complex128.pow(a.re, b);
        } else {
            result = exp(ln(a).mult(b));
        }

        return result;
    }


    /**
     * Computes the exponential function with the given input.
     * @param exponent Input to the exponential function.
     * @return The output of the exponential function.
     */
    public static Complex128 exp(Complex128 exponent) {
        double expRe = Math.exp(exponent.re);
        double re = expRe*Math.cos(exponent.im);
        double im = expRe*Math.sin(exponent.im);

        return new Complex128(re, im);
    }


    /**
     * Computes the natural logarithm of a double. For non-negative values this function is equivalent to {@link Math#log}.
     * If the number is negative, then it is passed on as a complex value to {@link Complex128#ln(Complex128)}.
     * @param num Input to the complex natural logarithm function.
     * @return The principle value of the complex natural logarithm for the given input.
     */
    public static Complex128 ln(double num) {
        Complex128 result;

        if(num<0) {
            result = Complex128.ln(new Complex128(num));
        } else {
            result = new Complex128(Math.log(num));
        }

        return result;
    }


    /**
     * Computes the complex natural logarithm of a complex number. This function is the analytic continuation of
     * the natural logarithm, that is the log base {@link Math#E e}.
     * @param num Input to the complex natural logarithm function.
     * @return The principle value of the complex natural logarithm for the given input.
     */
    public static Complex128 ln(Complex128 num) {
        Complex128 result;

        if(num.isReal() && num.re >=0) {
            result = new Complex128(Math.log(num.re));

        } else {
            double re = Math.log(Math.sqrt(num.re*num.re + num.im*num.im));
            double im = Math.atan2(num.im, num.re);

            result = new Complex128(re, im);
        }

        return result;
    }


    /**
     * Computes the complex logarithm base 10 of a complex number. Please note, this is <b>NOT</b> the natural logarithm.
     * If the complex natural logarithm is desired see {@link #ln(double)}. To specify a base, see {@link #log(double, Complex128)}
     * or {@link #log(Complex128, Complex128)}. If the argument is non-negative, then this function is equivalent to {@link Math#log(double)}.
     * @param num Input to the complex logarithm base 10 function.
     * @return The principle value of the complex logarithm base 10 for the given input.
     */
    public static Complex128 log(double num) {
        Complex128 result;

        if(num <= 0) {
            result = Complex128.log(new Complex128(num));
        } else {
            result = new Complex128(Math.log10(num));
        }

        return result;
    }


    /**
     * Computes the complex logarithm base 10 of a complex number. Please note, this is <b>NOT</b> the complex natural logarithm.
     * If the complex natural logarithm is desired see {@link #ln(Complex128)}. To specify a base, see {@link #log(double, Complex128)}
     * or {@link #log(Complex128, Complex128)}.
     * @param num Input to the complex logarithm base 10 function.
     * @return The principle value of the complex logarithm base 10 for the given input.
     */
    public static Complex128 log(Complex128 num) {
        Complex128 result;

        if(num.isReal() && num.re >=0) {
            result = new Complex128(Math.log10(num.re));
        } else {
            // Using the change of base formula
            Complex128 numerator = ln(num);
            Complex128 denominator = new Complex128(Math.log(10));

            result = numerator.div(denominator);
        }

        return result;
    }


    /**
     * Computes the complex logarithm, with specified base, of a complex number.
     * @param base Base of the logarithm.
     * @param num Input to the complex logarithm function with specified base.
     * @return The principle value of the complex logarithm, with specified base, for the given input.
     */
    public static Complex128 log(double base, double num) {
        return Complex128.log(new Complex128(base), new Complex128(num));
    }


    /**
     * Computes the complex logarithm, with specified base, of a complex number.
     * @param base Base of the logarithm.
     * @param num Input to the complex logarithm function with specified base.
     * @return The principle value of the complex logarithm, with specified base, for the given input.
     */
    public static Complex128 log(double base, Complex128 num) {
        return Complex128.log(new Complex128(base), num);
    }


    /**
     * Computes the complex logarithm, with specified base, of a complex number.
     * @param base Base of the logarithm.
     * @param num Input to the complex logarithm function with specified base.
     * @return The principle value of the complex logarithm, with specified base, for the given input.
     */
    public static Complex128 log(Complex128 base, Complex128 num) {
        // Using the change of base formula
        Complex128 numerator = ln(num);
        Complex128 denominator = ln(base);

        return numerator.div(denominator);
    }


    /**
     * Computes the principle square root of a number. This method wraps {@link Math#sqrt(double)} and returns a {@link Complex128}.
     * @param num Input to square root.
     * @return The principle square root of {@code num}.
     */
    public static Complex128 sqrt(double num) {
        Complex128 result;

        if(num >=0) {
            result = new Complex128(Math.sqrt(num));
        } else {
            result = new Complex128(0, Math.sqrt(-num));
        }

        return result;
    }


    /**
     * Converts a complex number to an equivalent polar from.
     * @return An array of length 2 containing in order, the radius and angle (in radians) if the complex number.
     */
    public double[] toPolar() {
        double[] polar = new double[2];

        // Compute the magnitude and angle of the complex number.
        polar[0] = this.mag();
        polar[1] = Math.atan2(this.im, this.re);

        return polar;
    }


    /**
     * Converts a complex number expressed in polar from to the rectangular form.
     * @param r Radius of complex number.
     * @param theta Angle of the complex number in radians.
     * @return An equivalent complex number in rectangular form.
     */
    public static Complex128 fromPolar(double r, double theta) {
        double re = r*Math.cos(theta);
        double im = r*Math.sin(theta);

        return new Complex128(re, im);
    }


    /**
     * Computes the 2 argument arc-tangent function for a complex number. That is, for a complex number a+bi, atan2(b, a)
     * is computed. This method wraps {@link Math#atan2(double, double)}. <br>
     * To get the result as an {@link Complex128} see {@link Complex128#atan2AsComplex128(Complex128)}.
     * @param num The input to the atan2 function.
     * @return The output of the atan2 function given the specified input. If the complex number is zero, then {@link Double#NaN}
     * is returned.
     */
    public static double atan2(Complex128 num) {
        return Math.atan2(num.im, num.re);
    }


    /**
     * Computes the complex argument function for a complex number.
     * is computed. This method is equivalent to {@link Complex128#atan2(Complex128)}. <br>
     * To get the result as an {@link Complex128} see {@link Complex128#argAsComplex128(Complex128)}.
     * @param num The input to the atan2 function.
     * @return The output of the atan2 function given the specified input. If the complex number is zero, then {@link Double#NaN}
     * is returned.
     */
    public static double arg(Complex128 num) {
        return atan2(num);
    }


    /**
     * Computes the complex argument function for a complex number.
     * is computed. This method wraps is equivalent to {@link Complex128#atan2AsComplex128(Complex128) Complex128}. <br>
     * To get the result as a double see {@link Complex128#arg(Complex128)}.
     * @param num The input to the atan2 function.
     * @return The output of the atan2 function given the specified input. If the complex number is zero, then {@link Double#NaN}
     * is returned.
     */
    public static Complex128 argAsComplex128(Complex128 num) {
        return atan2AsComplex128(num);
    }


    /**
     * Computes the 2 argument arc-tangent function for a complex number. That is, for a complex number a+bi, atan2(b, a)
     * is computed. This method wraps {@link Math#atan2(double, double)}. <br>
     * To get the result as a double see {@link Complex128#atan2(Complex128)}.
     * @param num The input to the atan2 function.
     * @return The output of the atan2 function given the specified input. If the complex number is zero, then {@link Double#NaN}
     * is returned.
     */
    public static Complex128 atan2AsComplex128(Complex128 num) {
        return new Complex128(Math.atan2(num.im, num.re));
    }


    /**
     * Computes the trigonometric sine of a complex value.
     * @param num Complex valued input to the sine function. If num is real, then this is an angle in radians.
     * @return The trigonometric sine function evaluated at the specified value.
     */
    public static Complex128 sin(Complex128 num) {
        double re = Math.sin(num.re)*Math.cosh(num.im);
        double im = Math.cos(num.re)*Math.sinh(num.im);

        return new Complex128(re, im);
    }


    /**
     * Computes the inverse sine of a complex value. <br>
     * @param num Input to the inverse sine function.
     * @return The inverse sine of the input value.
     */
    public static Complex128 asin(Complex128 num) {
        Complex128 diff = ONE.sub(num.mult(num));
        Complex128 denominator = diff.sqrt();

        return atan(num.div(denominator));
    }


    /**
     * Computes the hyperbolic sine of a complex value.
     * @param num Input to the hyperbolic sine function.
     * @return The hyperbolic sine of the input value.
     */
    public static Complex128 sinh(Complex128 num) {
        Complex128 numerator = exp(num).add(exp(num.addInv()));
        Complex128 denominator = new Complex128(2);

        return numerator.div(denominator);
    }


    /**
     * Computes the trigonometric cosine value of a complex value.
     * @param num Complex valued input to the cosine function. If num is real, then this is an angle in radians.
     * @return The trigonometric cosine function evaluated at the specified value.
     */
    public static Complex128 cos(Complex128 num) {
        double re = Math.cos(num.re)*Math.cosh(num.im);
        double im = Math.sin(num.re)*Math.sinh(num.im);

        return new Complex128(re, im);
    }


    /**
     * Computes the inverse cosine of a complex value. <br>
     * @param num Input to the inverse cosine function.
     * @return The inverse cosine of the input value.
     */
    public static Complex128 acos(Complex128 num) {
        Complex128 term = new Complex128(Math.PI/2);

        return term.sub(asin(num));
    }


    /**
     * Computes the hyperbolic cosine of a complex value.
     * @param num Input to the hyperbolic cosine function.
     * @return The hyperbolic cosine of the input value.
     */
    public static Complex128 cosh(Complex128 num) {
        Complex128 numerator = exp(num).sub(exp(num.addInv()));
        Complex128 denominator = new Complex128(2);

        return numerator.div(denominator);
    }


    /**
     * Computes the trigonometric tangent value of a complex value.
     * @param num Complex valued input to the tangent function. If num is real, then this is an angle in radians.
     * @return The trigonometric tangent function evaluated at the specified value.
     */
    public static Complex128 tan(Complex128 num) {
        Complex128 numerator = new Complex128(Math.tan(num.re),
                Math.tanh(num.im));
        Complex128 denominator = new Complex128(1,
                -Math.tan(num.re)*Math.tanh(num.im));

        return numerator.div(denominator);
    }


    /**
     * Computes the inverse tangent of a complex value. <br>
     * @param num Input to the inverse tangent function.
     * @return The inverse tangent of the input value.
     */
    public static Complex128 atan(Complex128 num) {
        Complex128 factor = TWO.multInv();
        Complex128 numerator = IMAGINARY_UNIT.sub(num);
        Complex128 denominator = IMAGINARY_UNIT.add(num);
        Complex128 log = Complex128.ln(numerator.div(denominator));

        return factor.mult(log);
    }


    /**
     * Computes the hyperbolic tangent of a complex value.
     * @param num Input to the hyperbolic tangent function.
     * @return The hyperbolic tangent of the input value.
     */
    public static Complex128 tanh(Complex128 num) {
        Complex128 exp = exp(TWO.mult(num));
        Complex128 numerator = exp.sub(ONE);
        Complex128 denominator = exp.add(ONE);

        return numerator.div(denominator);
    }


    /**
     * The complex signum function. Please note, if the value passed to this method is zero, the function will
     * return zero.
     *
     * @param value Value to evaluate the signum function at.
     * @return If the number is zero then this function returns zero. Otherwise, returns the number divided by its magnitude.
     */
    public static Complex128 sgn(Complex128 value) {

        if(value.equals(ZERO)) {
            return value;
        } else {
            double magnitude = value.mag();
            return new Complex128(value.re / magnitude, value.im / magnitude);
        }
    }


    /**
     * Rounds both components of a complex number to the nearest respective integer.
     * @param n The complex number to round.
     * @return A complex number with integer real and imaginary components closest to the real and imaginary
     * components of the parameter n
     * @throws NumberFormatException If n is {@link Double#NaN}, {@link Double#POSITIVE_INFINITY} or
     * {@link Double#NEGATIVE_INFINITY}
     * @see #round(Complex128, int)
     */
    public static Complex128 round(Complex128 n) {
        return round(n, 0);
    }


    /**
     * Rounds number to specified number of decimal places. If the number is complex,
     * both the real and imaginary parts will be rounded.
     *
     * @param n Number to round.
     * @param decimals Number of decimals to round to.
     * @return The number <code>n</code> rounded to the specified
     * 		number of decimals.
     * @throws IllegalArgumentException If decimals is less than zero.
     * @throws NumberFormatException If n is {@link Double#NaN}, {@link Double#POSITIVE_INFINITY} or
     * {@link Double#NEGATIVE_INFINITY}
     * @see #round(Complex128)
     */
    public static Complex128 round(Complex128 n, int decimals) {
        if (decimals < 0) {
            throw new IllegalArgumentException(ErrorMessages.getNegValueErr(decimals));
        }

        double real;
        double imaginary;

        if(Double.isFinite(n.re)) {
            real = BigDecimal.valueOf(n.re).setScale(decimals, RoundingMode.HALF_UP).doubleValue();
        } else {
            real = n.re;
        }

        if(Double.isFinite(n.im)) {
            imaginary = BigDecimal.valueOf(n.im).setScale(decimals, RoundingMode.HALF_UP).doubleValue();
        } else {
            imaginary = n.im;
        }

        return new Complex128(real, imaginary);
    }


    /**
     * Rounds a complex numbers to zero if its magnitude within the specified tolerance from zero.
     * @param n Number to round.
     * @param tol Max distances in complex plane for which number should be rounded to zero.
     * @return The
     */
    public static Complex128 roundToZero(Complex128 n, double tol) {
        if(nearZero(n, tol)) {
            return ZERO;
        } else {
            return n;
        }
    }


    /**
     * Checks if a number is near zero in magnitude.
     * @param n Number to round.
     * @param tol Tolerance of how close to zero is
     * 		considered "near".
     * @return Returns true if magnitude of number is less than or equal to
     * 		<code>tol</code>. Otherwise, returns false.
     * @throws IllegalArgumentException If tol is less than 0.
     */
    public static boolean nearZero(Complex128 n, double tol) {
        if (tol < 0) {
            throw new IllegalArgumentException(ErrorMessages.getNegValueErr(tol));
        }

        return n.mag() <= tol;
    }


    /**
     * Compares this element of the field with {@code b}.
     * @param b Second elemetn of the field.
     * @return An int value:
     * <ul>
     *     <li>0 if this field element is equal to {@code b} in magnitude.</li>
     *     <li>< 0 if this field element is less than {@code b} in magnitude.</li>
     *     <li>> 0 if this field element is greater than {@code b} in magnitude.</li>
     *     Hence, this method returns zero if and only if the two field elemetns are equal in magnitude, a negative value if and
     *     only the field element it was called on is less than {@code b} in magnitude and positive if and only if the field
     *     element it was called on is greater than {@code b} in magnitude.
     * </ul>
     */
    public int compareTo(Complex128 b) {
        return Double.compare(this.mag(), b.mag());
    }


    /**
     * Computes the minimum magnitude from an array of complex numbers.
     * @param values Array of values to compute the minimum magnitude from.
     * @return The minimum magnitude from the {@code values array}. If the array has zero length, then -1 is returned.
     */
    public static Complex128 min(Complex128... values) {
        double min = -1;
        double currMag;

        if(values.length > 0) {
            min = values[0].mag();
        }

        for(Complex128 value : values) {
            currMag = value.mag();
            if(currMag < min) {
                min = currMag;
            }
        }

        return new Complex128(min);
    }


    /**
     * Computes the minimum real component from an array of complex numbers. All imaginary components are ignored.
     * @param values Array of values to compute the minimum real component from.
     * @return The minimum magnitude from the {@code values array}. If the array has zero length, {@link Double#NaN} is
     * returned.
     */
    public static Complex128 minRe(Complex128... values) {
        double min = Double.MAX_VALUE;
        double currMin;

        if(values.length == 0) {
            min = Double.NaN;
        }

        for(Complex128 value : values) {
            currMin = value.re;
            if(currMin < min) {
                min = currMin;
            }
        }

        return new Complex128(min);
    }


    /**
     * Computes the maximum magnitude from an array of complex numbers.
     * @param values Array of values to compute the maximum magnitude from.
     * @return The minimum magnitude from the {@code values array}. If the array has zero length, then -1 is returned.
     */
    public static Complex128 max(Complex128... values) {
        double max = -1;
        double currMax;

        if(values.length > 0) {
            max = values[0].mag();
        }

        for(Complex128 value : values) {
            currMax = value.mag();
            if(currMax > max) {
                max = currMax;
            }
        }

        return new Complex128(max);
    }


    /**
     * Computes the minimum real component from an array of complex numbers. All imaginary components are ignored.
     * @param values Array of values to compute the minimum real component from.
     * @return The minimum magnitude from the {@code values array}. If the array has zero length, {@link Double#NaN} is
     * returned.
     */
    public static Complex128 maxRe(Complex128... values) {
        double max = Double.MIN_NORMAL;
        double currMax;

        if(values.length == 0) {
            max = Double.NaN;
        }

        for(Complex128 value : values) {
            currMax = value.re;
            if(currMax > max) {
                max = currMax;
            }
        }

        return new Complex128(max);
    }


    /**
     * Computes the index of the minimum magnitude from an array of complex numbers.
     * @param values Array of values to compute the index of the minimum magnitude from.
     * @return The index of the minimum magnitude from the {@code values array}. If the array has zero length, then -1 is returned.
     */
    public static int argmin(Complex128... values) {
        double min = -1;
        double currMag;
        int arg = -1;

        if(values.length > 0) {
            min = values[0].mag();
        }

        for(int i=0; i<values.length; i++) {
            currMag = values[i].mag();
            if(currMag < min) {
                min = currMag;
                arg = i;
            }
        }

        return arg;
    }


    /**
     * Computes the index of the minimum real component from an array of complex numbers. All imaginary components are ignored.
     * @param values Array of values to compute the index of the minimum real component from.
     * @return The index of the minimum magnitude from the {@code values array}. If the array has zero length, -1 is returned.
     */
    public static int argminReal(Complex128... values) {
        double min = Double.MAX_VALUE;
        double currMin;
        int arg = -1;

        if(values.length == 0) {
            min = Double.NaN;
        }

        for(int i=0; i<values.length; i++) {
            currMin = values[i].re;
            if(currMin < min) {
                min = currMin;
                arg = i;
            }
        }

        return arg;
    }


    /**
     * Computes the index of the maximum magnitude from an array of complex numbers.
     * @param values Array of values to compute the index of the maximum magnitude from.
     * @return The index of the minimum magnitude from the {@code values array}. If the array has zero length, then -1 is returned.
     */
    public static int argmax(Complex128... values) {
        double max = -1;
        double currMag;
        int arg = -1;

        if(values.length > 0) {
            max = values[0].mag();
        }

        for(int i=0; i<values.length; i++) {
            currMag = values[i].mag();
            if(currMag > max) {
                max = currMag;
                arg = i;
            }
        }

        return arg;
    }


    /**
     * Computes the index of the minimum real component from an array of complex numbers. All imaginary components are ignored.
     * @param values Array of values to compute the index of the minimum real component from.
     * @return The index of the minimum magnitude from the {@code values array}. If the array has zero length, -1 is
     * returned.
     */
    public static int argmaxReal(Complex128... values) {
        double max = Double.MIN_NORMAL;
        double currMax;
        int arg = -1;

        if(values.length == 0) {
            max = Double.NaN;
        }

        for (Complex128 value : values) {
            currMax = value.re;
            if (currMax > max) {
                max = currMax;
            }
        }

        return arg;
    }


    /**
     * Checks if this complex number is a real valued integer.
     * @return True if the real component of this number is an integer and the complex component is zero. Otherwise, returns false.
     */
    public boolean isInt() {
        boolean result = !(isInfinite() || isNaN());
        return Math.rint(re)==re && im==0 && result;
    }


    /**
     * Checks if this complex number is a real valued double.
     * @return True if the complex component is zero. Otherwise, returns false.
     */
    public boolean isDouble() {
        return im==0;
    }


    /**
     * <p>Checks if this field element is finite in magnitude.</p>
     * <p>NOTE: It may be possible for this method to return true and {@link #mag()} to return {@link Double#POSITIVE_INFINITY},
     * {@link Double#NEGATIVE_INFINITY}, or {@link Double#NaN} if both components are finite but computing the magnitude explicitly
     * results in an overflow. </p>
     * @return True if this field element is finite in magnitude. False otherwise (i.e. infinite, NaN etc.).
     */
    public boolean isFinite() {
        // If both components are finite then the complex number will have finite magnitude.
        return Double.isFinite(re) && Double.isFinite(im);
    }


    /**
     * Checks if this field element is infinite in magnitude.
     * @return True if this field element is infinite in magnitude. False otherwise (i.e. finite, NaN, etc.).
     */
    public boolean isInfinite() {
        // If either components is infinite then the complex number will have infinite magnitude.
        return Double.isInfinite(re) || Double.isInfinite(im);
    }


    /**
     * Checks if this field element is NaN in magnitude.
     * @return True if this field element is NaN in magnitude. False otherwise (i.e. finite, NaN, etc.).
     */
    @Override
    public boolean isNaN() {
        return Double.isNaN(re) || Double.isNaN(im);
    }


    /**
     * Gets the real component of this complex number.
     * @return The real component of this complex number.
     */
    public double re() {
        return re;
    }


    /**
     * Gets the imaginary component of this complex number.
     * @return The imaginary component of this complex number.
     */
    public double im() {
        return im;
    }


    /**
     * Gets the length of the string representation of this complex number.
     * @param a Complex number.
     * @return The length of the string representation of the number.
     */
    public static int length(Complex128 a) {
        return a.toString().length();
    }


    /**
     * Converts the complex number to a string representation.
     * @return A string representation of the complex number.
     */
    @Override
    public String toString() {
        StringBuilder result = new StringBuilder();

        if (re != 0) {
            result.append(re % 1 == 0 ? (int) re : re);
        }

        if (im != 0) {
            if (im > 0 && re != 0) {
                result.append(" + ");
            } else if (im < 0) {
                result.append(" - ");
            }

            double absIm = Math.abs(im);
            if (absIm != 1) {
                result.append(absIm % 1 == 0 ? (int) absIm : absIm);
            }
            result.append("i");
        }

        if (re == 0 && im == 0) {
            return "0";
        }

        return result.toString();
    }
}
