// Affichage du PDF
function showPDF(event, pdfUrl) {
    event.preventDefault();
    const pdfViewer = document.getElementById('pdfViewer');
    const pdfFrame = document.getElementById('pdfFrame');
    pdfFrame.src = pdfUrl;
    pdfViewer.style.display = 'flex';
}

function hidePDF() {
    const pdfViewer = document.getElementById('pdfViewer');
    const pdfFrame = document.getElementById('pdfFrame');
    pdfFrame.src = '';
    pdfViewer.style.display = 'none';
}


// Affichage des informations de contact
function showContactInfo() {
    const contactInfo = document.getElementById('contactInfo');
    if (contactInfo.style.display === 'block') {
        contactInfo.style.display = 'none';
    } else {
        contactInfo.style.display = 'block';
    }
}

document.addEventListener('click', function(event) {
    const contactInfo = document.getElementById('contactInfo');
    const contactElement = document.querySelector('.contact');

    if (!contactElement.contains(event.target) && !contactInfo.contains(event.target)) {
        contactInfo.style.display = 'none';
    }
});

// Affichage des détails du projet
function toggleDetails(event, projectName = '', title = '', description = '', link = '') {
    event.stopPropagation();
    const projectDetails = document.getElementById('projectDetails');
    const detailsProjectName = document.getElementById('projectName');
    const detailsTitle = document.getElementById('detailsTitle');
    const detailsDescription = document.getElementById('detailsDescription');
    const detailsLink = document.getElementById('detailsLink');

    if (projectDetails.style.display === 'block') {
        projectDetails.style.display = 'none';
    } else {
        detailsProjectName.textContent = projectName;
        detailsTitle.textContent = title;
        detailsDescription.innerHTML = description; // Accept HTML content
        detailsLink.href = link;
        projectDetails.style.display = 'block';
    }
}

// Boutons dépliables
document.addEventListener("DOMContentLoaded", function() {
    var collapsibles = document.getElementsByClassName("collapsible");
    for (var i = 0; i < collapsibles.length; i++) {
        collapsibles[i].addEventListener("click", function() {
            this.classList.toggle("active");
            var content = this.nextElementSibling;
            if (content.classList.contains('show')) {
                content.classList.remove('show');
            } else {
                content.classList.add('show');
            }
        });
    }
});



