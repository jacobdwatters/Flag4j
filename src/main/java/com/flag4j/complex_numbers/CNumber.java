package com.flag4j.complex_numbers;

/**
 * Complex Number.
 */
public class CNumber extends Number {
    // Several constants are provided for convenience.
    /**
     * The complex number with zero imaginary and real parts.
     */
    public final static CNumber ZERO = new CNumber();
    /**
     * The complex number with zero imaginary part and one real part.
     */
    public final static CNumber ONE = new CNumber(1);
    /**
     * The complex number with zero imaginary part and negative one real part.
     */
    public final static CNumber NEGATIVE_ONE = new CNumber(1);
    /**
     * The real double value closer to pi than any other.
     */
    public final static CNumber PI = new CNumber(Math.PI);
    /**
     * The real double value closer to the mathematical constant e than any other.
     */
    public final static CNumber E = new CNumber(Math.E);
    /**
     * The imaginary unit i.
     */
    public final static CNumber IMAGINARY_UNIT = new CNumber(0, 1);
    /**
     * The additive inverse of the imaginary unit, -i.
     */
    public final static CNumber INV_IMAGINARY_UNIT = new CNumber(0, -1);
    /**
     * The maximum real double value 1.7976931348623157e308.
     */
    public final static CNumber MAX_REAL = new CNumber(Double.MAX_VALUE);
    /**
     * The minimum real double value 4.9E-324
     */
    public final static CNumber MIN_REAL = new CNumber(Double.MIN_VALUE);
    /**
     * The smallest possible real normal double 2.2250738585072014E-308.
     */
    public final static CNumber MIN_REAL_NORMAL = new CNumber(Double.MIN_NORMAL);
    /**
     * Complex number with real part equal to {@link Double#POSITIVE_INFINITY}.
     */
    public final static CNumber POSITIVE_INFINITY = new CNumber(Double.POSITIVE_INFINITY);
    /**
     * Complex number with real part equal to {@link Double#NEGATIVE_INFINITY}.
     */
    public final static CNumber NEGATIVE_INFINITY = new CNumber(Double.NEGATIVE_INFINITY);
    /**
     * Complex number with real part equal to {@link Double#NaN}.
     */
    public final static CNumber NaN = new CNumber(Double.NaN);

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
     * Constructs a complex number from a string of the form "a +/- bi" where and b are real values and either may be
     * omitted. i.e. "a", "bi", "a +/- i", and "i" are all also valid.
     * @param num The string representation of a complex number.
     */
    public CNumber(String num) {
        CNumber complexNum = CNumberParser.parseNumber(num);
        this.re = complexNum.re;
        this.im = complexNum.im;
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
     * Subtracting two complex numbers.
     * @param b The number to subtract from complex number.
     * @return The result of subtracting b from this complex number.
     */
    public CNumber sub(CNumber b) {
        return new CNumber(this.re - b.re, this.im - b.im);
    }


    /**
     * Checks if two complex numbers are equal. That is, if both numbers have equivalent real and complex parts.
     * @param b The object to compare.
     * @return True if b is a complex number and is equivalent to this complex number in both the real and
     * imaginary components. False, otherwise.
     */
    @Override
    public boolean equals(Object b) {
        boolean result = false;

        if(b instanceof CNumber) {
            CNumber bCopy = (CNumber) b;

            if(this.re==bCopy.re && this.im==bCopy.im) {
                result = true;
            }
        }

        return result;
    }


    /**
     * Generates the hashcode for this CNumber.
     * @return
     */
    @Override
    public int hashCode() {
        return Double.hashCode(re) + Double.hashCode(im);
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

}
