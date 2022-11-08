package com.flag4j.concurrency;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ErrorMessages;

import java.util.Random;


/**
 * This class contains several built in operations for use in the {@link ConcurrentOperation}.
 */
public final class BuiltInOperations {

    private BuiltInOperations() {
        // Hide constructor in the utility class
        throw new IllegalStateException(ErrorMessages.utilityClassErrMsg());
    }

    private static Random rand = new Random();
    // TODO: Javadoc

    public static final BinaryOperation<Double> addReal = Double::sum;
    public static final BinaryOperation<Double> subReal = (Double a, Double b) -> a-b;
    public static final BinaryOperation<Double> multReal = (Double a, Double b) -> a*b;
    public static final BinaryOperation<Double> divReal = (Double a, Double b) -> a/b;

    public static final BinaryOperation<CNumber> addComplex = CNumber::add;
    public static final BinaryOperation<CNumber> subComplex = CNumber::sub;
    public static final BinaryOperation<CNumber> multComplex = CNumber::mult;
    public static final BinaryOperation<CNumber> divComplex = CNumber::div;

    public static final UnaryOperation<Double> roundReal = (Double a) -> Double.valueOf(Math.round(a));
    public static final UnaryOperation<CNumber> roundComplex = (CNumber a) -> new CNumber(Math.round(a.re), Math.round(a.im));

    public static final ZeroOperation<Double> randnReal = () -> rand.nextGaussian();
    public static final ZeroOperation<Double> randuReal = () -> Math.random();

    public static final ZeroOperation<CNumber> randnComplex = () -> new CNumber(rand.nextGaussian(), rand.nextGaussian());
    public static final ZeroOperation<CNumber> randuComplex = () -> new CNumber(Math.random(), Math.random());
}
