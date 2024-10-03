package org.flag4j.sparse_vector;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.flag4j.util.exceptions.TensorShapeException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class CooVectorReshapeTests {

    static CooVector a;
    CooVector exp;

    @BeforeAll
    static void setup() {
        double[] values = {1.34, 51.6, -0.00245};
        int[] indices = {0, 5, 103};
        int size = 304;
        a = new CooVector(size, values, indices);
    }

    @Test
    void reshapeTestCase() {
        double[] values = {1.34, 51.6, -0.00245};
        int[] indices = {0, 5, 103};
        int size = 304;

        // -------------------- Sub-case 1 --------------------
        exp = new CooVector(size, values, indices);
        assertEquals(exp, a.reshape(new Shape(size)));

        // -------------------- Sub-case 2 --------------------
        exp = new CooVector(size, values, indices);
        assertEquals(exp, a.reshape(size));

        // -------------------- Sub-case 3 --------------------
        exp = new CooVector(size, values, indices);
        assertEquals(exp, a.flatten());

        // -------------------- Sub-case 4 --------------------
        exp = new CooVector(size, values, indices);
        assertEquals(exp, a.flatten(0));

        // -------------------- Sub-case 5 --------------------
        assertThrows(TensorShapeException.class, ()->a.reshape(new Shape(size-3)));

        // -------------------- Sub-case 6 --------------------
        assertThrows(TensorShapeException.class, ()->a.reshape(size-3));

        // -------------------- Sub-case 7 --------------------
        assertThrows(LinearAlgebraException.class, ()->a.flatten(1));
    }
}
