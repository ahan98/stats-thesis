use itertools::Itertools;

fn main() {
    let (n, k) = (5, 2);
    let bitmask = format!("{:0<width$}", "1".repeat(k), width=n);
    println!("{}", bitmask);
    let string = "hello";
    println!("{}", string);
    print_type_of(&bitmask);

    // let it = "110".chars().permutations(3);
    // for combo in it {
    //     println!("{:?}", combo);
    // }
    // println!("{}", it);

    let list: [i32; 3] = [1,2,3];
    let combos = comb(&list, 2);
    println!("{:?}", combos);
}

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}


fn comb<T>(slice: &[T], bitmask: usize) -> Vec<Vec<T>>
where
    T: Copy,
{
    // If k == 1, return a vector containing a vector for each element of the slice.
    if k == 1 {
        return slice.iter().map(|x| vec![*x]).collect::<Vec<Vec<T>>>();
    }
    // If k is exactly the slice length, return the slice inside a vector.
    if k == slice.len() {
        return vec![slice.to_vec()];
    }

    // Make a vector from the first element + all combinations of k - 1 elements of the rest of the slice.
    let mut result = comb(&slice[1..], k - 1)
        .into_iter()
        .map(|x| [&slice[..1], x.as_slice()].concat())
        .collect::<Vec<Vec<T>>>();

    // Extend this last vector with the all the combinations of k elements after from index 1 onward.
    result.extend(comb(&slice[1..], k));
    // Return final vector.
    return result;
}

