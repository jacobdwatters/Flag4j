package com.flag4j.complex_numbers;

/**
 * Complex Number.
 */
public class CNumber extends Number {
    /**
     * The complex number with zero imaginary and real parts.
     */
    final static CNumber ZERO = new CNumber();
    /**
     * The complex number with zero imaginary part and one real part.
     */
    final static CNumber ONE = new CNumber(1);

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
     * Constructs a complex number with specified complex and real components.
     * @param re Real component of complex number.
     * @param im Imaginary component of complex number.
     */
    public CNumber(double re, double im) {
        this.re = re;
        this.im = im;
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
     * Creates a new complex number which is the copy of the specified complex number.
     * @param a The complex number to copy.
     */
    public CNumber(CNumber a) {
        this.re = a.re;
        this.im = a.im;
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
}
