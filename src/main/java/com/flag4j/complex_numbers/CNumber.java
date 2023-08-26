/*
 * MIT License
 *
 * Copyright (c) 2022 Jacob Watters
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

package com.flag4j.complex_numbers;

import com.flag4j.util.ErrorMessages;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * A complex number stored in rectangular form. The real and imaginary components of the complex number are stored as
 * 64 bit doubles.
 */
public class CNumber extends Number {
    // Several constants are provided for convenience.
    /**
     * The complex number with zero imaginary and real parts.
     */
    private static final CNumber ZERO = new CNumber();
    /**
     * The complex number with zero imaginary part and one real part.
     */
    private static final CNumber ONE = new CNumber(1);
    /**
     * The complex number with zero imaginary part and two real part.
     */
    private static final CNumber TWO = new CNumber(2);
    /**
     * The complex number with zero imaginary part and negative one real part.
     */
    private static final CNumber NEGATIVE_ONE = new CNumber(1);
    /**
     * The real double value closer to pi than any other.
     */
    private static final CNumber PI = new CNumber(Math.PI);
    /**
     * The real double value closer to the mathematical constant e than any other.
     */
    private static final CNumber E = new CNumber(Math.E);
    /**
     * The double value closer than any other to the square root of 2
     */
    private static final CNumber ROOT_TWO = new CNumber(Math.sqrt(2));
    /**
     * The double value closer than any other to the square root of 3
     */
    private static final CNumber ROOT_THREE = new CNumber(Math.sqrt(3));
    /**
     * The imaginary unit i.
     */
    private static final CNumber IMAGINARY_UNIT = new CNumber(0, 1);
    /**
     * The additive inverse of the imaginary unit, -i.
     */
    private static final CNumber INV_IMAGINARY_UNIT = new CNumber(0, -1);
    /**
     * The maximum real double value 1.7976931348623157E308.
     */
    private static final CNumber MAX_REAL = new CNumber(Double.MAX_VALUE);
    /**
     * The minimum real double value 4.9E-324
     */
    private static final CNumber MIN_REAL = new CNumber(Double.MIN_VALUE);
    /**
     * The smallest possible real normal double 2.2250738585072014E-308.
     */
    private static final CNumber MIN_REAL_NORMAL = new CNumber(Double.MIN_NORMAL);
    /**
     * Complex number with real part equal to {@link Double#POSITIVE_INFINITY}.
     */
    private static final CNumber POSITIVE_INFINITY = new CNumber(Double.POSITIVE_INFINITY);
    /**
     * Complex number with real part equal to {@link Double#NEGATIVE_INFINITY}.
     */
    private static final CNumber NEGATIVE_INFINITY = new CNumber(Double.NEGATIVE_INFINITY);
    /**
     * Complex number with real and imaginary parts equal to {@link Double#NaN}.
     */
    private static final CNumber NaN = new CNumber(Double.NaN, Double.NaN);

    /**
     * Real component of the complex number.
     */
    public double re;
    /**
     * Imaginary component of the complex number.
     */
    public double im;


    /**
     * Constructs a complex number with value and magnitude 0.
     */
    public CNumber() {
        re = 0;
        im = 0;
    }


    /**
     * Constructs a complex number with specified real component and zero imaginary component.
     * @param re Real component of complex number.
     */
    public CNumber(double re) {
        this.re = re;
        this.im = 0;
    }


    /**
     * Constructs a complex number with specified complex and real components.
     * @param re Real component of complex number.
     * @param im Imaginary component of complex number.
     */
    public CNumber(double re, double im) {
        this.re = re;
        this.im = im;
    }


    /**
     * Creates a new complex number which is the copy of the specified complex number.
     * @param num The complex number to copy.
     */
    public CNumber(CNumber num) {
        this.re = num.re;
        this.im = num.im;
    }


    /**
     * Constructs a complex number from a string of the form {@code "a +/- bi"} where {@code a} and {b} are real values and either may be
     * omitted. i.e. {@code "a", "bi", "a +/- i"}, and {@code "i"} are all also valid.
     * @param num The string representation of a complex number.
     */
    public CNumber(String num) {
        CNumber complexNum = CNumberParser.parseNumber(num);
        this.re = complexNum.re;
        this.im = complexNum.im;
    }
    


    /**
     * Creates a copy of this complex number. Same as {@link #CNumber(CNumber)}.
     * @return A complex number with real and complex components equivalent to this complex number.
     */
    public CNumber copy() {
        return new CNumber(this);
    }


    /**
     * Checks if two complex numbers are equal. That is, if both numbers have equivalent real and complex parts.
     * @param b The object to compare.
     * @return True if: <br>
     * - {@code b} is a complex number and is equivalent to this complex number in both the real and
     * imaginary components. <br>
     * - This complex number has zero imaginary component and {@code b} is a {@link Float} or {@link Double} equal to this complex numbers
     * real component. <br>
     * - This complex number has zero imaginary component and {@code b} is a {@link Byte}, {@link Short}, {@link Integer},
     * or {@link Long} equal to this complex numbers
     * real component. <br>
     * Otherwise, returns false.
     */
    @Override
    public boolean equals(Object b) {
        boolean result = false;

        if(b instanceof CNumber) {
            CNumber num = (CNumber) b;
            result = re==num.re && im==num.im;
        } else if(b instanceof Number) {
            double num = ((Number) b).doubleValue();
            result = re==num && im==0;
        }

        return result;
    }


    /**
     * Checks if a complex number is equal to some double value. That is, if the real component of this complex number
     * is zero and the real component is equivalent to the double parameter.
     * @param b The double to compare.
     * @return True if b is a complex number and is equivalent to this complex number in both the real and
     * imaginary components. False, otherwise.
     */
    public boolean equals(double b) {
        return this.im==0 && this.re==b;
    }


    /**
     * Generates the hashcode for this CNumber.
     * @return An integer hash for this CNumber.
     */
    @Override
    public int hashCode() {
        final int hashPrime1 = 7;
        final int hashPrime2 = 31;

        int hash = hashPrime2*hashPrime1 + Double.hashCode(this.re);
        hash = hashPrime2*hash + Double.hashCode(this.im);
        return hash;
    }


    /**
     * Gets the value of the specified number as an {@code int}. This will be calculated only with the real component of
     * this {@link CNumber}.
     *
     * @return the numeric value represented by this object after conversion
     * to type {@code int}.
     */
    @Override
    public int intValue() {
        return (int) re;
    }


    /**
     * Gets the value of the specified number as a {@code long}. This will be calculated only with the real component of
     * this {@link CNumber}.
     *
     * @return the numeric value represented by this object after conversion
     * to type {@code long}.
     */
    @Override
    public long longValue() {
        return (long) re;
    }


    /**
     * Gets the value of the specified number as a {@code float}. This will be calculated only with the real component of
     * this {@link CNumber}.
     *
     * @return the numeric value represented by this object after conversion
     * to type {@code float}.
     */
    @Override
    public float floatValue() {
        return (float) re;
    }


    /**
     * Gets the value of the specified number as a {@code double}. This will be calculated only with the real component of
     * this {@link CNumber}.
     *
     * @return the numeric value represented by this object after conversion
     * to type {@code double}.
     */
    @Override
    public double doubleValue() {
        return re;
    }


    /**
     * Gets the double value of the imaginary component of this complex number.
     * @return imaginary part of given Number as double
     */
    public double doubleImaginaryValue() {
        return im;
    }


    /**
     * Note: This method may result in loss of accuracy
     *
     * @return imaginary part of given Number as float
     */
    public float floatImaginaryValue() {
        return (float) im;
    }

    /**
     * Note: This method may result in loss of accuracy
     *
     * @return imaginary part of given Number as int
     */
    public int intImaginaryValue() {
        return (int) im;
    }


    /**
     * Note: This method may result in loss of accuracy
     *
     * @return imaginary part of given Number as long
     */
    public long longImaginaryValue() {
        return (long) im;
    }


    /**
     * Adds two complex numbers.
     * @param b The number to add to this complex number.
     * @return The result of adding this complex number with b.
     */
    public CNumber add(CNumber b) {
        return new CNumber(this.re + b.re, this.im + b.im);
    }


    /**
     * Adds a double to a complex number.
     * @param b The double value to add to this complex number.
     * @return The result of adding b to this complex number.
     */
    public CNumber add(double b) {
        return new CNumber(this.re + b, this.im);
    }


    /**
     * Adds a specified number to this complex number and stores the result in this complex number.
     * @param b The value to add to this complex number.
     */
    public void addEq(CNumber b) {
        this.re += b.re;
        this.im += b.im;
    }


    /**
     * Adds a specified number to this complex number and stores the result in this complex number.
     * @param b The value to add to this complex number.
     */
    public void addEq(double b) {
        this.re += b;
    }


    /**
     * Subtracts a specified number from this complex number and stores the result in this complex number.
     * @param b The value to add to this complex number.
     */
    public void subEq(CNumber b) {
        this.re -= b.re;
        this.im -= b.im;
    }


    /**
     * Subtracts a specified number to this complex number and stores the result in this complex number.
     * @param b The value to add to this complex number.
     */
    public void subEq(double b) {
        this.re -= b;
    }


    /**
     * Subtracting two complex numbers.
     * @param b The number to subtract from complex number.
     * @return The result of subtracting b from this complex number.
     */
    public CNumber sub(CNumber b) {
        return new CNumber(this.re - b.re, this.im - b.im);
    }


    /**
     * subtracts a double from a complex number.
     * @param b The double value to subtract from this complex number.
     * @return The result of subtracting b from this complex number.
     */
    public CNumber sub(double b) {
        return new CNumber(this.re - b, this.im);
    }


    /**
     * Sums an array of complex numbers.
     * @param numbers Numbers to sum.
     * @return The sum of all values in numbers.
     */
    public static CNumber sum(CNumber... numbers) {
        double re = 0;
        double im = 0;

        for(CNumber num : numbers) {
               re += num.re;
               im += num.im;
        }

        return new CNumber(re, im);
    }


    /**
     * Computes the multiplication of two complex numbers.
     *
     * @param b Second complex number in the product.
     * @return Product of this complex number with b.
     */
    public CNumber mult(CNumber b) {
        return new CNumber(
                this.re*b.re - this.im*b.im,
                this.re*b.im + this.im*b.re
        );
    }


    /**
     * Computes the multiplication of a complex number and a double.
     *
     * @param b Second complex number in the product.
     * @return Product of this complex number with b.
     */
    public CNumber mult(double b) {
        return new CNumber(this.re*b, this.im*b);
    }


    /**
     * Multiplies this complex number with another complex number and stores the result in this {@link CNumber}.
     *
     * @param b Second complex number in the product.
     */
    public void multEq(CNumber b) {
        this.re = this.re*b.re - this.im*b.im;
        this.im = this.re*b.im + this.im*b.re;
    }


    /**
     * Multiplies this complex number with another complex number and stores the result in this {@link CNumber}.
     *
     * @param b Second complex number in the product.
     */
    public void multEq(double b) {
        this.re *= b;
        this.im *= b;
    }


    /**
     * Computes the division of two complex numbers.
     * @param b The divisor for the complex division.
     * @return The quotient of this complex number with b. If b is equivalent to {@link CNumber#ZERO} then
     *  {@link CNumber#NaN} is returned.
     */
    public CNumber div(CNumber b) {
        CNumber quotient;

        if (this.equals(ZERO) && !b.equals(ZERO)) {
            quotient = new CNumber();

        } else if(b.isReal() && !b.equals(ZERO)) {
            quotient = new CNumber(this.re/b.re, this.im/b.re);

        } else {
            double divisor = b.re*b.re + b.im*b.im;

            quotient = new CNumber(
                    (this.re*b.re + this.im*b.im) / divisor,
                    (this.im*b.re - this.re*b.im) / divisor);
        }

        return quotient;
    }


    /**
     * Computes the division of a complex numbers with a double value.
     * @param b The divisor for the complex division.
     * @return The quotient of this complex number with b. If b is zero then
     *  {@link CNumber#NaN} is returned.
     */
    public CNumber div(double b) {
        CNumber quotient;

        if (this.equals(ZERO) && b != 0) {
            quotient = new CNumber();
        }
        else {
            quotient = new CNumber(this.re/b, this.im/b);
        }

        return quotient;
    }


    /**
     * Computes the absolute value / magnitude of a complex number. <br>
     * Note: This method is the same as {@link #mag()}.
     * @return The absolute value/magnitude of this complex number.
     */
    public double abs() {
        return mag();
    }


    /**
     * Computes the magnitude value of a complex number as a double. <br>
     * @return The absolute value/magnitude of this complex number as a double.
     */
    public double mag() {
        return Math.sqrt(this.re*this.re + this.im*this.im);
    }


    /**
     * Computes the additive inverse of this complex number.
     * @return The additive inverse of this complex number.
     */
    public CNumber addInv() {
        return new CNumber(-this.re, -this.im);
    }


    /**
     * Computes the multiplicative inverse of this complex number.
     * @return The multiplicative inverse of this complex number.
     */
    public CNumber multInv() {
        return ONE.div(this);
    }


    /**
     * Computes the complex conjugate of this complex number.
     * @return The complex conjugate of this complex number.
     */
    public CNumber conj() {
        return new CNumber(this.re, -this.im);
    }


    /**
     * Compute a raised to the power of b. This method wraps {@link Math#pow(double, double)}
     * and returns a {@link CNumber}.
     * @param a The base.
     * @param b The exponent.
     * @return a raised to the power of b.
     */
    public static CNumber pow(double a, double b) {
        return new CNumber(Math.pow(a, b));
    }


    /**
     * Compute a raised to the power of b.
     * and returns a {@link CNumber}.
     * @param a The base.
     * @param b The exponent.
     * @return a to the power of b.
     */
    public static CNumber pow(double a, CNumber b) {
        CNumber result;

        if(b.im == 0) {
            result = CNumber.pow(a, b.re);
        } else {
            // Apply a change of base
            double logA = Math.log(a);
            CNumber power = new CNumber(logA * b.re, logA * b.im);
            result = exp(power);
        }

        return result;
    }


    /**
     * Compute a raised to the power of b.
     * and returns a {@link CNumber}.
     * @param a The base.
     * @param b The exponent.
     * @return a to the power of b.
     */
    public static CNumber pow(CNumber a, CNumber b) {
        CNumber result;

        if(a.im == 0) {
            if(b.im == 0) {
                result = CNumber.pow(a.re, b.re);
            } else {
                result = CNumber.pow(a.re, b);
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
    public static CNumber pow(CNumber a, double b) {
        CNumber result;

        if(a.im == 0) {
            result = CNumber.pow(a.re, b);
        } else {
            result = exp(ln(a).mult(b));
        }

        return result;
    }


    /**
     * Computes the exponential function with the given input. This method simply wraps the {@link Math#exp(double)}
     * method and returns a {@link CNumber}.
     * @param exponent Input to the exponential function.
     * @return The output of the exponential function.
     */
    public static CNumber exp(double exponent) {
        double re = Math.exp(exponent);
        return new CNumber(re);
    }


    /**
     * Computes the exponential function with the given input.
     * @param exponent Input to the exponential function.
     * @return The output of the exponential function.
     */
    public static CNumber exp(CNumber exponent) {
        double expRe = Math.exp(exponent.re);
        double re = expRe*Math.cos(exponent.im);
        double im = expRe*Math.sin(exponent.im);

        return new CNumber(re, im);
    }


    /**
     * Computes the natural logarithm of a double. For non-negative values this function is equivalent to {@link Math#log}.
     * If the number is negative, then it is passed on as a complex value to {@link CNumber#ln(CNumber)}.
     * @param num Input to the complex natural logarithm function.
     * @return The principle value of the complex natural logarithm for the given input.
     */
    public static CNumber ln(double num) {
        CNumber result;

        if(num<0) {
            result = CNumber.ln(new CNumber(num));
        } else {
            result = new CNumber(Math.log(num));
        }

        return result;
    }


    /**
     * Computes the complex natural logarithm of a complex number. This function is the analytic continuation of
     * the natural logarithm, that is the log base {@link Math#E e}.
     * @param num Input to the complex natural logarithm function.
     * @return The principle value of the complex natural logarithm for the given input.
     */
    public static CNumber ln(CNumber num) {
        CNumber result;

        if(num.isReal() && num.re >=0) {
            result = new CNumber(Math.log(num.re));

        } else {
            double re = Math.log(Math.sqrt(num.re*num.re + num.im*num.im));
            double im = Math.atan2(num.im, num.re);

            result = new CNumber(re, im);
        }

        return result;
    }


    /**
     * Computes the complex logarithm base 10 of a complex number. Please note, this is <b>NOT</b> the natural logarithm.
     * If the complex natural logarithm is desired see {@link #ln(double)}. To specify a base, see {@link #log(double, CNumber)}
     * or {@link #log(CNumber, CNumber)}. If the argument is non-negative, then this function is equivalent to {@link Math#log(double)}.
     * @param num Input to the complex logarithm base 10 function.
     * @return The principle value of the complex logarithm base 10 for the given input.
     */
    public static CNumber log(double num) {
        CNumber result;

        if(num <= 0) {
            result = CNumber.log(new CNumber(num));
        } else {
            result = new CNumber(Math.log10(num));
        }

        return result;
    }


    /**
     * Computes the complex logarithm base 10 of a complex number. Please note, this is <b>NOT</b> the complex natural logarithm.
     * If the complex natural logarithm is desired see {@link #ln(CNumber)}. To specify a base, see {@link #log(double, CNumber)}
     * or {@link #log(CNumber, CNumber)}.
     * @param num Input to the complex logarithm base 10 function.
     * @return The principle value of the complex logarithm base 10 for the given input.
     */
    public static CNumber log(CNumber num) {
        CNumber result;

        if(num.isReal() && num.re >=0) {
            result = new CNumber(Math.log10(num.re));
        } else {
            // Using the change of base formula
            CNumber numerator = ln(num);
            CNumber denominator = new CNumber(Math.log(10));

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
    public static CNumber log(double base, double num) {
        return CNumber.log(new CNumber(base), new CNumber(num));
    }


    /**
     * Computes the complex logarithm, with specified base, of a complex number.
     * @param base Base of the logarithm.
     * @param num Input to the complex logarithm function with specified base.
     * @return The principle value of the complex logarithm, with specified base, for the given input.
     */
    public static CNumber log(double base, CNumber num) {
        return CNumber.log(new CNumber(base), num);
    }


    /**
     * Computes the complex logarithm, with specified base, of a complex number.
     * @param base Base of the logarithm.
     * @param num Input to the complex logarithm function with specified base.
     * @return The principle value of the complex logarithm, with specified base, for the given input.
     */
    public static CNumber log(CNumber base, CNumber num) {
        // Using the change of base formula
        CNumber numerator = ln(num);
        CNumber denominator = ln(base);

        return numerator.div(denominator);
    }


    /**
     * Computes the principle square root of a number. This method wraps {@link Math#sqrt(double)} and returns a {@link CNumber}.
     * @param num Input to square root.
     * @return The principle square root of {@code num}.
     */
    public static CNumber sqrt(double num) {
        CNumber result;

        if(num >=0) {
            result = new CNumber(Math.sqrt(num));
        } else {
            result = new CNumber(0, Math.sqrt(-num));
        }

        return result;
    }


    /**
     * Computes the principle square root of a number.
     * @param num Input to square root.
     * @return The principle square root of {@code num}. That is, the square root of {@code num} with positive real part.
     */
    public static CNumber sqrt(CNumber num) {
        CNumber result;

        if(num.isReal() && num.re>=0) {
            result = new CNumber(Math.sqrt(num.re));

        } else if(num.isReal() && num.re<0) {
            result = new CNumber(0, Math.sqrt(-num.re));

        } else {
            double mag = num.mag();
            double factor = num.im / Math.abs(num.im);

            double re = Math.sqrt((mag + num.re)/2);
            double im = factor*Math.sqrt((mag - num.re)/2);

            result = new CNumber(re, im);
        }

        return result;
    }


    /**
     * The complex signum function. Please note, if the value passed to this method is zero, the function will
     * return zero.
     *
     * @param value Value to evaluate the signum function at.
     * @return If the number is zero then this function returns zero. Otherwise, returns the number divided by its magnitude.
     */
    public static CNumber sgn(CNumber value) {
        CNumber result;

        if(value.equals(CNumber.ZERO)) {
            result = new CNumber();

        } else if(value.im == 0) {
            if(value.re > 0) {
                result = new CNumber(1);
            } else {
                result = new CNumber(-1);
            }
        } else {
            result = value.div(value.mag());
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
    public static CNumber fromPolar(double r, double theta) {
        double re = r*Math.cos(theta);
        double im = r*Math.sin(theta);

        return new CNumber(re, im);
    }


    /**
     * Computes the 2 argument arc-tangent function for a complex number. That is, for a complex number a+bi, atan2(b, a)
     * is computed. This method wraps {@link Math#atan2(double, double)}. <br>
     * To get the result as an {@link CNumber} see {@link CNumber#atan2AsCNumber(CNumber)}.
     * @param num The input to the atan2 function.
     * @return The output of the atan2 function given the specified input. If the complex number is zero, then {@link Double#NaN}
     * is returned.
     */
    public static double atan2(CNumber num) {
        return Math.atan2(num.im, num.re);
    }


    /**
     * Computes the complex argument function for a complex number.
     * is computed. This method is equivalent to {@link CNumber#atan2(CNumber)}. <br>
     * To get the result as an {@link CNumber} see {@link CNumber#argAsCNumber(CNumber)}.
     * @param num The input to the atan2 function.
     * @return The output of the atan2 function given the specified input. If the complex number is zero, then {@link Double#NaN}
     * is returned.
     */
    public static double arg(CNumber num) {
        return atan2(num);
    }


    /**
     * Computes the complex argument function for a complex number.
     * is computed. This method wraps is equivalent to {@link CNumber#atan2AsCNumber(CNumber) CNumber}. <br>
     * To get the result as a double see {@link CNumber#arg(CNumber)}.
     * @param num The input to the atan2 function.
     * @return The output of the atan2 function given the specified input. If the complex number is zero, then {@link Double#NaN}
     * is returned.
     */
    public static CNumber argAsCNumber(CNumber num) {
        return atan2AsCNumber(num);
    }


    /**
     * Computes the 2 argument arc-tangent function for a complex number. That is, for a complex number a+bi, atan2(b, a)
     * is computed. This method wraps {@link Math#atan2(double, double)}. <br>
     * To get the result as a double see {@link CNumber#atan2(CNumber)}.
     * @param num The input to the atan2 function.
     * @return The output of the atan2 function given the specified input. If the complex number is zero, then {@link Double#NaN}
     * is returned.
     */
    public static CNumber atan2AsCNumber(CNumber num) {
        return new CNumber(Math.atan2(num.im, num.re));
    }


    /**
     * Computes the trigonometric sine of a value. <br>
     * Note, this method wraps {@link Math#sin(double)} and returns a {@link CNumber}.
     * @param num Input angle in radians.
     * @return The trigonometric sine function evaluated at the specified value.
     */
    public static CNumber sin(double num) {
        return new CNumber(Math.sin(num));
    }


    /**
     * Computes the trigonometric sine of a complex value.
     * @param num Complex valued input to the sine function. If num is real, then this is an angle in radians.
     * @return The trigonometric sine function evaluated at the specified value.
     */
    public static CNumber sin(CNumber num) {
        double re = Math.sin(num.re)*Math.cosh(num.im);
        double im = Math.cos(num.re)*Math.sinh(num.im);

        return new CNumber(re, im);
    }


    /**
     * Computes the inverse sine of a value. <br>
     * Note, this method wraps {@link Math#asin(double)} and returns {@link CNumber}.
     * @param num Input to the inverse sine function.
     * @return The inverse sine of the input value. That is, the angle whose sine value is num.
     */
    public static CNumber asin(double num) {
        return new CNumber(Math.asin(num));
    }


    /**
     * Computes the inverse sine of a complex value. <br>
     * @param num Input to the inverse sine function.
     * @return The inverse sine of the input value.
     */
    public static CNumber asin(CNumber num) {
        CNumber diff = ONE.sub(num.mult(num));
        CNumber denominator = CNumber.sqrt(diff);

        return atan(num.div(denominator));
    }


    /**
     * Computes the hyperbolic sine of a value. <br>
     * Note, this method wraps {@link Math#sinh(double)} and returns a {@link CNumber}.
     * @param num Input to the hyperbolic sine function.
     * @return The hyperbolic sine of the input value.
     */
    public static CNumber sinh(double num) {
        return new CNumber(Math.sinh(num));
    }


    /**
     * Computes the hyperbolic sine of a complex value.
     * @param num Input to the hyperbolic sine function.
     * @return The hyperbolic sine of the input value.
     */
    public static CNumber sinh(CNumber num) {
        CNumber numerator = exp(num).add(exp(num.addInv()));
        CNumber denominator = new CNumber(2);

        return numerator.div(denominator);
    }


    /**
     * Computes the trigonometric cosine of a value. <br>
     * Note, this method wraps {@link Math#cos(double)} and returns a {@link CNumber}.
     * @param num Input angle in radians.
     * @return The trigonometric cosine function evaluated at the specified value.
     */
    public static CNumber cos(double num) {
        return new CNumber(Math.cos(num));
    }


    /**
     * Computes the trigonometric cosine value of a complex value.
     * @param num Complex valued input to the cosine function. If num is real, then this is an angle in radians.
     * @return The trigonometric cosine function evaluated at the specified value.
     */
    public static CNumber cos(CNumber num) {
        double re = Math.cos(num.re)*Math.cosh(num.im);
        double im = Math.sin(num.re)*Math.sinh(num.im);

        return new CNumber(re, im);
    }


    /**
     * Computes the inverse cosine of a value. <br>
     * Note, this method wraps {@link Math#acos(double)} and returns {@link CNumber}.
     * @param num Input to the inverse cosine function.
     * @return The inverse cosine of the input value. That is, the angle whose cosine value is num.
     */
    public static CNumber acos(double num) {
        return new CNumber(Math.acos(num));
    }


    /**
     * Computes the inverse cosine of a complex value. <br>
     * @param num Input to the inverse cosine function.
     * @return The inverse cosine of the input value.
     */
    public static CNumber acos(CNumber num) {
        CNumber term = new CNumber(Math.PI/2);

        return term.sub(asin(num));
    }


    /**
     * Computes the hyperbolic cosine of a value. <br>
     * Note, this method wraps {@link Math#cosh(double)} and returns a {@link CNumber}.
     * @param num Input to the hyperbolic cosine function.
     * @return The hyperbolic cosine of the input value.
     */
    public static CNumber cosh(double num) {
        return new CNumber(Math.cosh(num));
    }


    /**
     * Computes the hyperbolic cosine of a complex value.
     * @param num Input to the hyperbolic cosine function.
     * @return The hyperbolic cosine of the input value.
     */
    public static CNumber cosh(CNumber num) {
        CNumber numerator = exp(num).sub(exp(num.addInv()));
        CNumber denominator = new CNumber(2);

        return numerator.div(denominator);
    }


    /**
     * Computes the trigonometric tangent of a value. <br>
     * Note, this method wraps {@link Math#tan(double)} and returns a {@link CNumber}.
     * @param num Input angle in radians.
     * @return The trigonometric tangent function evaluated at the specified value.
     */
    public static CNumber tan(double num) {
        return new CNumber(Math.tan(num));
    }


    /**
     * Computes the trigonometric tangent value of a complex value.
     * @param num Complex valued input to the tangent function. If num is real, then this is an angle in radians.
     * @return The trigonometric tangent function evaluated at the specified value.
     */
    public static CNumber tan(CNumber num) {
        CNumber numerator = new CNumber(Math.tan(num.re),
                Math.tanh(num.im));
        CNumber denominator = new CNumber(1,
                -Math.tan(num.re)*Math.tanh(num.im));

        return numerator.div(denominator);
    }


    /**
     * Computes the inverse tangent of a value. <br>
     * Note, this method wraps {@link Math#atan(double)} and returns {@link CNumber}.
     * @param num Input to the inverse tangent function.
     * @return The inverse tangent of the input value. That is, the angle whose tangent value is num.
     */
    public static CNumber atan(double num) {
        return new CNumber(Math.atan(num));
    }


    /**
     * Computes the inverse tangent of a complex value. <br>
     * @param num Input to the inverse tangent function.
     * @return The inverse tangent of the input value.
     */
    public static CNumber atan(CNumber num) {
        CNumber factor = TWO.multInv();
        CNumber numerator = IMAGINARY_UNIT.sub(num);
        CNumber denominator = IMAGINARY_UNIT.add(num);
        CNumber log = CNumber.ln(numerator.div(denominator));

        return factor.mult(log);
    }


    /**
     * Computes the hyperbolic tangent of a value. <br>
     * Note, this method wraps {@link Math#tanh(double)} and returns a {@link CNumber}.
     * @param num Input to the hyperbolic tangent function.
     * @return The hyperbolic tangent of the input value.
     */
    public static CNumber tanh(double num) {
        return new CNumber(Math.tanh(num));
    }


    /**
     * Computes the hyperbolic tangent of a complex value.
     * @param num Input to the hyperbolic tangent function.
     * @return The hyperbolic tangent of the input value.
     */
    public static CNumber tanh(CNumber num) {
        CNumber exp = exp(TWO.mult(num));
        CNumber numerator = exp.sub(ONE);
        CNumber denominator = exp.add(ONE);

        return numerator.div(denominator);
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
     * Rounds both components of a complex number to the nearest respective integer.
     * @param n The complex number to round.
     * @return A complex number with integer real and imaginary components closest to the real and imaginary
     * components of the parameter n
     * @throws NumberFormatException If n is {@link Double#NaN}, {@link Double#POSITIVE_INFINITY} or
     * {@link Double#NEGATIVE_INFINITY}
     * @see #round(CNumber, int) 
     */
    public static CNumber round(CNumber n) {
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
     * @see #round(CNumber) 
     */
    public static CNumber round(CNumber n, int decimals) {
        if (decimals < 0) {
            throw new IllegalArgumentException(ErrorMessages.negValueErr(decimals));
        }

        double real, imaginary;

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

        return new CNumber(real, imaginary);
    }


    /**
     * Rounds a complex numbers to zero if its magnitude within the specified tolerance from zero.
     * @param n Number to round.
     * @param tol Max distances in complex plane for which number should be rounded to zero.
     * @return The
     */
    public static CNumber roundToZero(CNumber n, double tol) {
        if(nearZero(n, tol)) {
            return new CNumber();
        } else {
            return n.copy();
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
    public static boolean nearZero(CNumber n, double tol) {
        if (tol < 0) {
            throw new IllegalArgumentException(ErrorMessages.negValueErr(tol));
        }

        return n.mag() <= tol;
    }


    /**
     * Compares the size of two complex numbers (magnitudes).
     *
     * @param b Number to compare to this number.
     * @return
     * - If the magnitude of this number is equal to that of <code>b</code>, then this method will return 0. <br>
     * - If the magnitude of this number is less than that of <code>b</code>, then this method will return a negative number. <br>
     * - If the magnitude of this number is greater than that of <code>b</code>, then this method will return a positive number.
     */
    public int compareTo(CNumber b) {
        return Double.compare(this.mag(), b.mag());
    }


    /**
     * Compares the real value of two numbers.
     *
     * @param b Number to compare to this number.
     * @return
     * - If the real value of this number is equal to that of <code>b</code>, then this method will return 0. <br>
     * - If the real value of this number is less than that of <code>b</code>, then this method will return a negative number. <br>
     * - If the real value of this number is greater than that of <code>b</code>, then this method will return a positive number.
     */
    public int compareToReal(CNumber b) {
        return compareToReal(b.re);
    }


    /**
     * Compares the real value of two numbers. This method wraps {@link Double#compareTo(Double)} which is computed
     * with the real component of this complex number.
     *
     * @param b Number to compare to this number.
     * @return
     * - If the real value of this number is equal to that of <code>b</code>, then this method will return 0. <br>
     * - If the real value of this number is less than that of <code>b</code>, then this method will return a negative number. <br>
     * - If the real value of this number is greater than that of <code>b</code>, then this method will return a positive number.
     */
    public int compareToReal(double b) {
        return Double.compare(this.re, b);
    }


    /**
     * Computes the minimum magnitude from an array of complex numbers.
     * @param values Array of values to compute the minimum magnitude from.
     * @return The minimum magnitude from the {@code values array}. If the array has zero length, then -1 is returned.
     */
    public static CNumber min(CNumber... values) {
        double min = -1;
        double currMag;

        if(values.length > 0) {
            min = values[0].mag();
        }

        for(CNumber value : values) {
            currMag = value.mag();
            if(currMag < min) {
                min = currMag;
            }
        }

        return new CNumber(min);
    }


    /**
     * Computes the minimum real component from an array of complex numbers. All imaginary components are ignored.
     * @param values Array of values to compute the minimum real component from.
     * @return The minimum magnitude from the {@code values array}. If the array has zero length, {@link Double#NaN} is
     * returned.
     */
    public static CNumber minRe(CNumber... values) {
        double min = Double.MAX_VALUE;
        double currMin;

        if(values.length == 0) {
            min = Double.NaN;
        }

        for(CNumber value : values) {
            currMin = value.re;
            if(currMin < min) {
                min = currMin;
            }
        }

        return new CNumber(min);
    }


    /**
     * Computes the maximum magnitude from an array of complex numbers.
     * @param values Array of values to compute the maximum magnitude from.
     * @return The minimum magnitude from the {@code values array}. If the array has zero length, then -1 is returned.
     */
    public static CNumber max(CNumber... values) {
        double max = -1;
        double currMax;

        if(values.length > 0) {
            max = values[0].mag();
        }

        for(CNumber value : values) {
            currMax = value.mag();
            if(currMax > max) {
                max = currMax;
            }
        }

        return new CNumber(max);
    }


    /**
     * Computes the minimum real component from an array of complex numbers. All imaginary components are ignored.
     * @param values Array of values to compute the minimum real component from.
     * @return The minimum magnitude from the {@code values array}. If the array has zero length, {@link Double#NaN} is
     * returned.
     */
    public static CNumber maxRe(CNumber... values) {
        double max = Double.MIN_NORMAL;
        double currMax;

        if(values.length == 0) {
            max = Double.NaN;
        }

        for(CNumber value : values) {
            currMax = value.re;
            if(currMax > max) {
                max = currMax;
            }
        }

        return new CNumber(max);
    }


    /**
     * Computes the index of the minimum magnitude from an array of complex numbers.
     * @param values Array of values to compute the index of the minimum magnitude from.
     * @return The index of the minimum magnitude from the {@code values array}. If the array has zero length, then -1 is returned.
     */
    public static int argMin(CNumber... values) {
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
    public static int argMinReal(CNumber... values) {
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
    public static int argMax(CNumber... values) {
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
    public static int argMaxReal(CNumber... values) {
        double max = Double.MIN_NORMAL;
        double currMax;
        int arg = -1;

        if(values.length == 0) {
            max = Double.NaN;
        }

        for (CNumber value : values) {
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
     * Checks if either component of this complex number is NaN.
     * @return True if either component is NaN. Otherwise, returns false.
     */
    public boolean isNaN() {
        return Double.isNaN(re) || Double.isNaN(im);
    }


    /**
     * Checks that both components of this complex number are finite valued.
     * @return True if both components are finite. Otherwise, returns false (including NaN).
     */
    public boolean isFinite() {
        return Double.isFinite(re) && Double.isFinite(im);
    }


    /**
     * Checks if either component of this complex number is infinitely large in absolute value.
     * @return True if either components are infinite. Otherwise, returns false.
     */
    public boolean isInfinite() {
        return Double.isInfinite(re) || Double.isInfinite(im);
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
    public static int length(CNumber a) {
        return a.toString().length();
    }


    /**
     * Gets the complex number equivalent to zero.
     * @return The complex number which is equivalent to zero.
     */
    public static CNumber zero() {
        return ZERO.copy();
    }


    /**
     * Gets the complex number equivalent to one.
     * @return The complex number equivalent to one.
     */
    public static CNumber one() {
        return ONE.copy();
    }


    /**
     * Gets the complex number equivalent to two.
     * @return The complex number equivalent to two.
     */
    public static CNumber two() {
        return TWO.copy();
    }


    /**
     * Gets the complex number equivalent to negative one.
     * @return The complex number equivalent to negative one.
     */
    public static CNumber negOne() {
        return NEGATIVE_ONE.copy();
    }


    /**
     * Gets the complex number equivalent to the square root of two.
     * @return The complex number equivalent to the square root of two.
     */
    public static CNumber rootTwo() {
        return ROOT_TWO.copy();
    }


    /**
     * Gets the complex number equivalent to the square root of three.
     * @return The complex number equivalent to the square root of three.
     */
    public static CNumber rootThree() {
        return ROOT_THREE.copy();
    }


    /**
     * Gets the complex number equivalent to {@link Double#POSITIVE_INFINITY}.
     * @return The complex number equivalent to {@link Double#POSITIVE_INFINITY}.
     */
    public static CNumber posInfinity() {
        return POSITIVE_INFINITY.copy();
    }


    /**
     * Gets the complex number equivalent to {@link Double#NEGATIVE_INFINITY}.
     * @return The complex number equivalent to {@link Double#NEGATIVE_INFINITY}.
     */
    public static CNumber negInfinity() {
        return NEGATIVE_INFINITY.copy();
    }


    /**
     * Gets the complex number equivalent to {@link Math#PI}.
     * @return The complex number equivalent to {@link Math#PI}.
     */
    public static CNumber pi() {
        return PI.copy();
    }


    /**
     * Gets the complex number equivalent to {@link Math#E}.
     * @return The complex number equivalent to {@link Math#E}.
     */
    public static CNumber eulers() {
        return E.copy();
    }


    /**
     * Gets the complex number equivalent to {@link Double#NaN}.
     * @return The complex number equivalent to {@link Double#NaN}.
     */
    public static CNumber nan() {
        return NaN.copy();
    }


    /**
     * Gets the complex number equivalent to {@code i}.
     * @return The complex number equivalent to {@code i}.
     */
    public static CNumber imagUnit() {
        return IMAGINARY_UNIT.copy();
    }


    /**
     * Gets the complex number equivalent to {@code -i}.
     * @return The complex number equivalent to {@code -i}.
     */
    public static CNumber negImagUnit() {
        return INV_IMAGINARY_UNIT.copy();
    }


    /**
     * Gets the complex number equivalent to {@link Double#MIN_VALUE}.
     * @return The complex number equivalent to {@link Double#MIN_VALUE}.
     */
    public static CNumber minReal() {
        return MIN_REAL.copy();
    }


    /**
     * Gets the complex number equivalent to {@link Double#MIN_NORMAL}.
     * @return The complex number equivalent to {@link Double#MIN_NORMAL}.
     */
    public static CNumber minRealNormal() {
        return MIN_REAL_NORMAL.copy();
    }


    /**
     * Gets the complex number equivalent to {@link Double#MAX_VALUE}.
     * @return The complex number equivalent to {@link Double#MAX_VALUE}.
     */
    public static CNumber maxReal() {
        return MAX_REAL.copy();
    }


    /**
     * Converts the complex number to a string representation.
     * @return A string representation of the complex number.
     */
    public String toString() {
        String result = "";

        double real = re, imaginary = im;

        if (real != 0) {
            if (real % 1 == 0) {
                result += (int) real;
            } else {
                result += real;
            }
        }

        if (imaginary != 0) {
            if (imaginary < 0 && real != 0) {
                result += " - ";
                imaginary = -imaginary;
            } else if (real != 0) {
                result += " + ";
            }

            if (imaginary % 1 == 0) {
                if(imaginary != 1) {
                    if(imaginary == -1) {
                        result += '-';
                    } else {
                        result += (int) imaginary;
                    }
                }
            } else {
                result += imaginary;
            }

            result += "i";
        }

        if (real == 0 && imaginary == 0) {
            result = "0";
        }

        return result;
    }
}
