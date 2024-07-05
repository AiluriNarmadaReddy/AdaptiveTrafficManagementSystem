function submitForm(action) {
    // Set the hidden input value to the selected action
    const actionInput = document.createElement('input');
    actionInput.type = 'hidden';
    actionInput.name = 'action';
    actionInput.value = action;
    document.getElementById('videoForm').appendChild(actionInput);

    // Submit the form
    document.getElementById('videoForm').submit();
}

document.getElementById('numLanes').addEventListener('input', function() {
    const numLanes = parseInt(this.value);
    const container = document.getElementById('videoInputsContainer');
    container.innerHTML = ''; // Clear previous content

    for (let i = 1; i <= numLanes; i++) {
        const div = document.createElement('div');
        div.classList.add('mb-3');

        const label = document.createElement('label');
        label.textContent = `Video for Lane ${i}:`;
        label.classList.add('form-label');

        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.name = `videoFile${i}`;
        fileInput.accept = 'video/*';
        fileInput.classList.add('form-control-file');
        fileInput.required = true;

        div.appendChild(label);
        div.appendChild(fileInput);
        container.appendChild(div);
    }
});
