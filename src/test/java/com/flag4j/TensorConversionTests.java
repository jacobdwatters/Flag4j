package com.flag4j;

import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TensorConversionTests {

    static double[] aEntries;
    static Shape shape, expShape;
    static Tensor A;
    static CTensor exp;
    static CNumber[] expEntries;

    @BeforeAll
    static void setup() {
        aEntries = new double[]{1.23, 2.556, -121.5, 15.61, 14.15, -99.23425,
                0.001345, 2.677, 8.14, -0.000194, 1, 234};
        shape = new Shape(1, 3, 2, 1, 2);
        A = new Tensor(shape, aEntries);
    }


    @Test
    void toComplexTest() {
        // --------------------- Sub-case 1 ---------------------
        expEntries = new CNumber[]{
                new CNumber(1.23), new CNumber(2.556), new CNumber(-121.5),
                new CNumber(15.61), new CNumber(14.15), new CNumber(-99.23425),
                new CNumber(0.001345), new CNumber(2.677), new CNumber(8.14),
                new CNumber(-0.000194), new CNumber(1), new CNumber(234)};
        exp = new CTensor(shape.copy(), expEntries);

        assertEquals(exp, A.toComplex());
    }


    @Test
    void toMatrixTest() {
        Tensor B;

        double[] expEntries;
        Shape expShape;
        Matrix exp;

        // ----------------------- Sub-case 1 -----------------------
        expEntries = aEntries.clone();
        expShape = new Shape(1, aEntries.length);
        exp = new Matrix(expShape, expEntries);

        assertEquals(exp, A.toMatrix());

        // ----------------------- Sub-case 2 -----------------------
        expEntries = aEntries.clone();
        expShape = new Shape(4, 3);
        B = A.reshape(expShape);
        exp = new Matrix(expShape, expEntries);

        assertEquals(exp, B.toMatrix());
    }


    @Test
    void toVectorTest() {
        double[] expEntries;
        Shape expShape;
        Vector exp;

        // ----------------------- Sub-case 1 -----------------------
        expEntries = aEntries.clone();
        exp = new Vector(expEntries);

        assertEquals(exp, A.toVector());
    }
}
