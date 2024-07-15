package org.flag4j.complex_tensor;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CTensor;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.util.ArrayUtils;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CTensorConversionTests {
    static CNumber[] aEntries;
    static Shape shape;
    static CTensor A;

    @BeforeAll
    static void setup() {
        aEntries = new CNumber[]{
                new CNumber(1.4415, -0.0245), new CNumber(235.61, 1.45), new CNumber(0, -0.00024),
                new CNumber(1.0), new CNumber(-85.1, 9.234), new CNumber(1.345, -781.2),
                new CNumber(0.014, -2.45),  new CNumber(-140.0),  new CNumber(0, 1.5),
                new CNumber(51.0, 24.56),  new CNumber(6.1, -0.03),  new CNumber(-0.00014, 1.34),};
        shape = new Shape(1, 3, 2, 1, 2);
        A = new CTensor(shape, aEntries);
    }


    @Test
    void toRealTestCase() {
        double[] expEntries;
        Tensor exp;

        // --------------------- Sub-case 1 ---------------------
        expEntries = new double[]{1.4415, 235.61, 0, 1.0, -85.1, 1.345, 0.014, -140.0, 0, 51.0, 6.1, -0.00014};
        exp = new Tensor(shape.copy(), expEntries);

        assertEquals(exp, A.toReal());
    }


    @Test
    void toMatrixTestCase() {
        CTensor B;

        CNumber[] expEntries;
        Shape expShape;
        CMatrix exp;

        // ----------------------- Sub-case 1 -----------------------
        expEntries = ArrayUtils.copyOf(aEntries);
        expShape = new Shape(1, aEntries.length);
        exp = new CMatrix(expShape, expEntries);

        assertEquals(exp, A.toMatrix());

        // ----------------------- Sub-case 2 -----------------------
        expEntries = ArrayUtils.copyOf(aEntries);
        expShape = new Shape(4, 3);
        B = A.reshape(expShape);
        exp = new CMatrix(expShape, expEntries);

        assertEquals(exp, B.toMatrix());
    }


    @Test
    void toVectorTestCase() {
        CNumber[] expEntries;
        CVector exp;

        // ----------------------- Sub-case 1 -----------------------
        expEntries = ArrayUtils.copyOf(aEntries);
        exp = new CVector(expEntries);

        assertEquals(exp, A.toVector());
    }
}
