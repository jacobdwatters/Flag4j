package org.flag4j.complex_tensor;

import org.flag4j.arrays.dense.CTensor;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.util.ArrayUtils;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CTensorReshapeTests {


    static CNumber[] aEntries;
    static Shape aShape, expShape;
    static CTensor A, exp;


    @BeforeAll
    static void setup() {
        aEntries = new CNumber[]{
                new CNumber(0.000123, 1.3445), new CNumber(5.23523, -0.351), new CNumber(0, -834513.36), new CNumber(235.6),
                new CNumber(934, 294.5), new CNumber(13.5, 0.00134), new CNumber(-0.0, 3.1), new CNumber(0.1, Double.NEGATIVE_INFINITY),
                new CNumber(-345, 1.4), new CNumber(-0.00134, -8345.6), new CNumber(1.00015, -0.425), new CNumber(Double.POSITIVE_INFINITY)};
        aShape = new Shape(3, 2, 2);
        A = new CTensor(aShape, aEntries);
    }


    @Test
    void reshapeTestCase() {
        // -------------------------- Sub-case 1 --------------------------
        expShape = new Shape(1, 1, 12, 1);
        exp = new CTensor(expShape, ArrayUtils.copyOf(aEntries));

        assertEquals(exp, A.reshape(expShape.copy()));

        // -------------------------- Sub-case 2 --------------------------
        expShape = new Shape(4, 3);
        exp = new CTensor(expShape, ArrayUtils.copyOf(aEntries));

        assertEquals(exp, A.reshape(expShape.copy()));

        // -------------------------- Sub-case 3 --------------------------
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, ArrayUtils.copyOf(aEntries));

        assertEquals(exp, A.reshape(expShape.copy()));

        // -------------------------- Sub-case 4 --------------------------
        expShape = new Shape(2, 2, 3, 1);
        exp = new CTensor(expShape, ArrayUtils.copyOf(aEntries));

        assertEquals(exp, A.reshape(expShape.copy()));
    }


    @Test
    void flattenTestCase() {
        // -------------------------- Sub-case 1 --------------------------
        expShape = new Shape(12);
        exp = new CTensor(expShape, ArrayUtils.copyOf(aEntries));

        assertEquals(exp, A.flatten());

        // -------------------------- Sub-case 2 --------------------------
        expShape = new Shape(1, 1, 12);
        exp = new CTensor(expShape, ArrayUtils.copyOf(aEntries));

        assertEquals(exp, A.flatten(2));

        // -------------------------- Sub-case 3 --------------------------
        expShape = new Shape(1, 12, 1);
        exp = new CTensor(expShape, ArrayUtils.copyOf(aEntries));

        assertEquals(exp, A.flatten(1));

        // -------------------------- Sub-case 4 --------------------------
        expShape = new Shape(12, 1, 1);
        exp = new CTensor(expShape, ArrayUtils.copyOf(aEntries));

        assertEquals(exp, A.flatten(0));

        // -------------------------- Sub-case 5 --------------------------
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.flatten(-1));

        // -------------------------- Sub-case 6 --------------------------
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.flatten(5));
    }
}
